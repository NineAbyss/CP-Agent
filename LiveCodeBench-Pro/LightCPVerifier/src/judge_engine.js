import fs from 'fs/promises';
import path from 'path';
import { toNs, toBytes, fileExists } from './utils.js';
import { GoJudgeClient } from './gojudge.js';
import { ProblemManager } from './problem_manager.js';

export class JudgeEngine {
    constructor(config) {
        this.problemManager = new ProblemManager({
            problemsRoot: config.problemsRoot,
            gjAddr: config.gjAddr,
            testlibPath: config.testlibPath
        });
        this.goJudge = new GoJudgeClient(config.gjAddr);
        this.submissionManager = config.submissionManager;
        this.testlibPath = config.testlibPath || '/lib/testlib';

        // In-memory queue and results
        this.queue = [];
        this.results = new Map();

        // Start worker threads
        this.startWorkers(config.workers || 4);
    }

    // Submit a task
    async submit(pid, lang, code) {
        const sid = await this.submissionManager.nextSubmissionId();
        this.results.set(sid, { status: 'queued' });
        const { bucketDir, subDir } = this.submissionManager.submissionPaths(sid);
        await fs.mkdir(bucketDir, { recursive: true });
        await fs.mkdir(subDir, { recursive: true });

        if (this.queue.length >= 1024 * 512) {
            this.queue.push({ sid, pid, lang });
            await fs.writeFile(
                path.join(subDir, `source.code`),
                code
            );
        } else {
            this.queue.push({ sid, pid, lang, code });
        }

        await fs.writeFile(
            path.join(subDir, 'meta.json'),
            JSON.stringify({ sid, pid, lang, ts: Date.now() }, null, 2)
        );

        return sid;
    }

    // Get result
    async getResult(sid) {
        const r = this.results.get(sid);
        if (r) {
            this.results.delete(sid);
            return r;
        }

        try {
            const { subDir } = this.submissionManager.submissionPaths(sid);
            const txt = await fs.readFile(path.join(subDir, 'result.json'), 'utf8');
            return JSON.parse(txt);
        } catch {
            return null;
        }
    }

    // Clear result cache
    clearResults() {
        this.results.clear();
    }

    // Judge a single test case
    async judgeCase({ runSpec, caseItem, problem, checkerId }) {
        // Read input/output files
        const inf = await this.problemManager.readTestFile(problem.pdir.split('/').pop(), caseItem.input);

        let ans;
        try {
            // Attempt to read .ans file
            const ansFile = caseItem.output.replace(/\.out$/, '.ans');
            ans = await this.problemManager.readTestFile(problem.pdir.split('/').pop(), ansFile);
        } catch {
            // Fallback to .out file if .ans is missing
            ans = await this.problemManager.readTestFile(problem.pdir.split('/').pop(), caseItem.output);
        }

        // Run student program
        const runRes = await this.goJudge.runOne({
            args: runSpec.runArgs,
            env: ['PATH=/usr/bin:/bin'],
            files: [{ content: inf }, { name: 'stdout', max: 128 * 1024 * 1024 }, { name: 'stderr', max: 64 * 1024 * 1024 }],
            cpuLimit: toNs(caseItem.time),
            clockLimit: toNs(caseItem.time) * 2,
            memoryLimit: toBytes(caseItem.memory),
            stackLimit: toBytes(caseItem.memory),
            addressSpaceLimit: true,
            procLimit: 128,
            copyIn: { ...runSpec.preparedCopyIn }
        });

        let extra = '';
        if (runRes.status === 'Signalled') {
            extra = `(signal=${runRes.error || 'unknown'})`;
        }

        if (runRes.status !== 'Accepted') {
            return {
                ok: false,
                status: runRes.status,
                time: runRes.runTime,
                memory: runRes.memory,
                msg: (runRes.files?.stderr || '') + extra
            };
        }

        const out = runRes.files?.stdout ?? '';

        // Run checker (testlib): chk in.txt out.txt ans.txt
        const chkRes = await this.goJudge.runOne({
            args: ['chk', 'in.txt', 'out.txt', 'ans.txt'],
            env: ['PATH=/usr/bin:/bin'],
            files: [{ content: '' }, { name: 'stdout', max: 1024 * 1024 }, { name: 'stderr', max: 1024 * 1024 }],
            cpuLimit: 10e9,
            clockLimit: 20e9,
            memoryLimit: 256 << 20,
            stackLimit: 256 << 20,
            procLimit: 128,
            copyIn: {
                'chk': { fileId: checkerId },
                'in.txt': { content: inf },
                'out.txt': { content: out },
                'ans.txt': { content: ans }
            }
        });

        console.log('Checker result:', chkRes);
        const ok = chkRes.status === 'Accepted' && chkRes.exitStatus === 0;
        return {
            ok,
            status: ok ? 'Accepted' : 'Wrong Answer',
            time: runRes.runTime,
            memory: runRes.memory,
            msg: chkRes.files?.stdout || chkRes.files?.stderr || ''
        };
    }

    async judgeInteractiveCase({ runSpec, caseItem, problem, interactorId }) {
        const inf = await this.problemManager.readTestFile(problem.pdir.split('/').pop(), caseItem.input);

        let ans;
        try {
            // Attempt to read .ans file
            const ansFile = caseItem.output.replace(/\.out$/, '.ans');
            ans = await this.problemManager.readTestFile(problem.pdir.split('/').pop(), ansFile);
        } catch {
            // Fallback to .out file if .ans is missing
            ans = await this.problemManager.readTestFile(problem.pdir.split('/').pop(), caseItem.output);
        }

        // Run student program with dual processes + bidirectional pipe
        const interactRes = await this.goJudge.run({
            cmd: [
                // index: 0 -> student program
                {
                    args: runSpec.runArgs,
                    env: ['PATH=/usr/bin:/bin'],
                    // Note: stdout is handled by the pipe, so set to null; keep stderr for debugging
                    files: [
                        null,                         // stdin (driven by interactor output)
                        null,                         // stdout (mapped via pipe)
                        { name: 'stderr', max: 1024 * 1024 }  // Capture stderr
                    ],
                    cpuLimit: toNs(caseItem.time),
                    clockLimit: toNs(caseItem.time) * 2,
                    memoryLimit: toBytes(caseItem.memory),
                    stackLimit: toBytes(caseItem.memory),
                    procLimit: 128,
                    copyIn: { ...runSpec.preparedCopyIn },
                    // Optional: working directory/uid limits etc.
                },

                // index: 1 -> interactor
                {
                    // Assuming interactor executable is named "interactor", typical pattern: interactor in.txt tout.txt ans.txt
                    args: ['interactor', 'in.txt', 'tout.txt', 'ans.txt'],
                    env: ['PATH=/usr/bin:/bin'],
                    // interactor reads from stdin (from student stdout) and writes to stdout (to student stdin)
                    files: [
                        null,                               // stdin (from student stdout pipe)
                        null,                               // stdout (mapped to student stdin)
                        { name: 'stderr', max: 1024 * 1024 }  // interactor stderr usually contains logs/verdicts
                    ],
                    cpuLimit: toNs(caseItem.time) * 4,
                    clockLimit: toNs(caseItem.time) * 4 * 2,
                    memoryLimit: toBytes(caseItem.memory) * 4,
                    stackLimit: toBytes(caseItem.memory) * 4,
                    procLimit: 128,
                    copyIn: {
                        'interactor': { fileId: interactorId }, // Add executable
                        'in.txt': { content: inf },             // Provide input to interactor
                        'ans.txt': { content: ans }             // Provide answer if needed
                    }
                }
            ],
            pipeMapping: [
                { in: { index: 0, fd: 1 }, out: { index: 1, fd: 0 } },
                { in: { index: 1, fd: 1 }, out: { index: 0, fd: 0 } }
            ]
        });

        const submissionRes = interactRes[0];
        const interactorRes = interactRes[1];

        if (submissionRes.status === 'Accepted' && interactorRes.status === 'Accepted'
            && interactorRes.exitStatus === 0 && submissionRes.exitStatus === 0) {
            return {
                ok: true,
                status: 'Accepted',
                time: submissionRes.runTime,
                memory: Math.max(submissionRes.memory, interactorRes.memory),
                msg: (interactorRes.files?.stdout || '') + (interactorRes.files?.stderr || '') // Combined logs
            };
        }
        if (submissionRes.status !== 'Accepted') {
            let extra = '';
            if (submissionRes.status === 'Signalled') {
                extra = ` (signal=${submissionRes.error || 'unknown'})`;
            }
            return {
                ok: false,
                status: submissionRes.status,
                time: submissionRes.runTime,
                memory: submissionRes.memory,
                msg: (submissionRes.files?.stderr || '') + extra
            };
        }
        if (interactorRes.status !== 'Accepted') {
            return {
                ok: false,
                status: interactorRes.status,
                time: submissionRes.runTime,
                memory: submissionRes.memory,
                msg: (interactorRes.files?.stderr || '')
            };
        }
    }

    // Start worker threads
    startWorkers(workerCount) {
        for (let i = 0; i < workerCount; i++) {
            this.startWorker();
        }
    }

    async judgeDefault(problem, sid, pid, lang, code, subDir) {
        let cleanupIds = [];
        let checkerCleanup, checkerId;
        try {
            // Prepare student program (compile/cache in sandbox)
            const runSpec = await this.goJudge.prepareProgram({
                lang,
                code,
                mainName: problem.filename || null
            });
            cleanupIds.push(...(runSpec.cleanupIds || []));


            // Read checker.bin file if exists
            const checkerBinPath = path.join(problem.pdir, `${problem.checker}.bin`);
            let checkerResult;
            if (await fileExists(checkerBinPath)) {
                checkerResult = await this.goJudge.copyInBin(checkerBinPath);
                checkerId = checkerResult.binId;
                checkerCleanup = checkerResult.cleanup;
            } else if (problem.checker) {
                // Otherwise read checker source and compile
                const chkSrc = await this.problemManager.readCheckerSource(pid, problem.checker);
                checkerResult = await this.goJudge.prepareChecker(chkSrc, this.testlibPath);
                checkerId = checkerResult.checkerId;
                checkerCleanup = checkerResult.cleanup;
            }

            // Iterate test cases (fail-fast on non-AC)
            const caseResults = [];
            let firstBad = null;
            for (const c of problem.cases) {
                const r = await this.judgeCase({ runSpec, caseItem: c, problem, checkerId });
                caseResults.push(r);
                if (!r.ok) {
                    firstBad = r;
                    break;
                }
            }
            const passed = firstBad === null;
            const result = caseResults[caseResults.length - 1].status || 'Unknown';

            const final = { status: 'done', passed, result, cases: caseResults };
            this.results.set(sid, final);
            await fs.writeFile(path.join(subDir, 'result.json'), JSON.stringify(final, null, 2));
        } catch (e) {
            const err = { status: 'error', error: String(e) };
            this.results.set(sid, err);
            await fs.writeFile(path.join(subDir, 'result.json'), JSON.stringify(err, null, 2));
        } finally {
            // Clean up go-judge cache files
            for (const id of cleanupIds) {
                await this.goJudge.deleteFile(id);
            }
            if (checkerCleanup) {
                await checkerCleanup();
            }
        }
    }

    async judgeInteractive(problem, sid, pid, lang, code, subDir) {
        let cleanupIds = [];
        let interactorCleanup, interactorId;
        try {
            // Prepare student program (compile/cache in sandbox)
            console.log('Preparing interactive program...');
            const runSpec = await this.goJudge.prepareProgram({
                lang,
                code,
                mainName: problem.filename || null
            });
            cleanupIds.push(...(runSpec.cleanupIds || []));

            // Read interactor.bin file if exists
            const interactorBinPath = path.join(problem.pdir, `${problem.interactor}.bin`);
            let interactorResult;
            if (await fileExists(interactorBinPath)) {
                interactorResult = await this.goJudge.copyInBin(interactorBinPath);
                interactorId = interactorResult.binId;
                interactorCleanup = interactorResult.cleanup;
            } else if (problem.interactor) {
                // Otherwise read interactor source and compile
                const interSrc = await this.problemManager.readInteractorSource(pid, problem.interactor);
                interactorResult = await this.goJudge.prepareInteractor(interSrc, this.testlibPath);
                interactorId = interactorResult.interactorId;
                interactorCleanup = interactorResult.cleanup;
            }

            // Iterate test cases (fail-fast on non-AC)
            const caseResults = [];
            let firstBad = null;
            for (const c of problem.cases) {
                const r = await this.judgeInteractiveCase({ runSpec, caseItem: c, problem, interactorId });
                caseResults.push(r);
                if (!r.ok) {
                    firstBad = r;
                    break;
                }
            }
            const passed = firstBad === null;
            const result = caseResults[caseResults.length - 1].status || 'Unknown';

            const final = { status: 'done', passed, result, cases: caseResults };
            this.results.set(sid, final);
            await fs.writeFile(path.join(subDir, 'result.json'), JSON.stringify(final, null, 2));
        } catch (e) {
            const err = { status: 'error', error: String(e) };
            this.results.set(sid, err);
            await fs.writeFile(path.join(subDir, 'result.json'), JSON.stringify(err, null, 2));
        } finally {
            // Clean up go-judge cache files
            for (const id of cleanupIds) {
                await this.goJudge.deleteFile(id);
            }
            if (interactorCleanup) {
                await interactorCleanup();
            }
        }
    }

    // Single worker thread loop
    async startWorker() {
        while (true) {
            const job = this.queue.shift();
            if (!job) {
                await new Promise(r => setTimeout(r, 50));
                continue;
            }

            let { sid, pid, lang, code } = job;
            const { bucketDir, subDir } = this.submissionManager.submissionPaths(sid);
            if (typeof code !== 'string') {
                code = await fs.readFile(path.join(subDir, 'source.code'), 'utf8');
            } else {
                await fs.writeFile(path.join(subDir, 'source.code'), code);
            }

            let problem;
            try {
                problem = await this.problemManager.loadProblem(pid);
            } catch (e) {
                const err = { status: 'error', error: `Load problem failed: ${e.message}` };
                this.results.set(sid, err);
                await fs.writeFile(path.join(subDir, 'result.json'), JSON.stringify(err, null, 2));
                continue;
            }

            switch (problem.cfg.type) {
                case 'interactive':
                    await this.judgeInteractive(problem, sid, pid, lang, code, subDir);
                    break;

                case 'leetcode':
                    throw new Error('LeetCode problems are not supported for now.');
                    break;

                default:
                    await this.judgeDefault(problem, sid, pid, lang, code, subDir);
                    break;
            }
        }
    }

    // Get source file name based on language
    getSourceFileName(lang) {
        switch (lang) {
            case 'cpp': return 'main.cpp';
            case 'py':
            case 'pypy': return 'main.py';
            case 'java': return 'Main.java';
            default: return 'main.txt';
        }
    }
}