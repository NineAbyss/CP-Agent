#include <iostream>
#include <vector>
#include <algorithm>
#include <cstring>
using namespace std;

const int MOD = 998244353;
const int G = 3; // primitive root

long long pow_mod(long long a, long long b) {
    long long res = 1;
    while (b) {
        if (b & 1) res = res * a % MOD;
        a = a * a % MOD;
        b >>= 1;
    }
    return res;
}

void ntt(vector<long long>& a, bool invert) {
    int n = a.size();
    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) swap(a[i], a[j]);
    }
    for (int len = 2; len <= n; len <<= 1) {
        long long wlen = pow_mod(G, (MOD - 1) / len);
        if (invert) wlen = pow_mod(wlen, MOD - 2);
        for (int i = 0; i < n; i += len) {
            long long w = 1;
            for (int j = 0; j < len / 2; j++) {
                long long u = a[i + j];
                long long v = a[i + j + len / 2] * w % MOD;
                a[i + j] = (u + v) % MOD;
                a[i + j + len / 2] = (u - v + MOD) % MOD;
                w = w * wlen % MOD;
            }
        }
    }
    if (invert) {
        long long inv_n = pow_mod(n, MOD - 2);
        for (long long& x : a) x = x * inv_n % MOD;
    }
}

vector<long long> convolution(vector<long long> const& a, vector<long long> const& b) {
    vector<long long> fa(a.begin(), a.end()), fb(b.begin(), b.end());
    int n = 1;
    while (n < (int)a.size() + (int)b.size() - 1) n <<= 1;
    fa.resize(n);
    fb.resize(n);
    ntt(fa, false);
    ntt(fb, false);
    for (int i = 0; i < n; i++) fa[i] = fa[i] * fb[i] % MOD;
    ntt(fa, true);
    fa.resize(a.size() + b.size() - 1);
    return fa;
}

// Fast IO
void read(int &x) {
    x = 0; char c = getchar();
    while (c < '0' || c > '9') c = getchar();
    while (c >= '0' && c <= '9') x = x * 10 + c - '0', c = getchar();
}
void read(long long &x) {
    x = 0; char c = getchar();
    while (c < '0' || c > '9') c = getchar();
    while (c >= '0' && c <= '9') x = x * 10 + c - '0', c = getchar();
}

int main() {
    // ios::sync_with_stdio(false);
    // cin.tie(0);
    int T;
    // cin >> T;
    read(T);
    while (T--) {
        int n, m;
        // cin >> n >> m;
        read(n); read(m);
        int nm = n * m;
        vector<long long> a(nm);
        for (int i = 0; i < nm; i++) read(a[i]);
        sort(a.begin(), a.end());
        vector<long long> b(nm + 1);
        b[0] = 0;
        for (int i = 1; i <= nm; i++) b[i] = a[i - 1];
        
        // factorials
        vector<long long> fact(nm + 1), invfact(nm + 1);
        fact[0] = 1;
        for (int i = 1; i <= nm; i++) fact[i] = fact[i - 1] * i % MOD;
        invfact[nm] = pow_mod(fact[nm], MOD - 2);
        for (int i = nm - 1; i >= 0; i--) invfact[i] = invfact[i + 1] * (i + 1) % MOD;
        
        // B array
        vector<long long> B(nm);
        for (int d = 0; d < nm; d++) {
            B[d] = (b[d + 1] - b[d] + MOD) % MOD * fact[d] % MOD;
        }
        
        // C array
        vector<long long> C(nm);
        for (int u = 0; u < nm; u++) {
            C[u] = invfact[u];
        }
        
        // convolution: D[k] = sum_{v=k}^{nm-1} B[v] * C[v-k]
        // Let A_rev[i] = B[nm-1-i]
        vector<long long> A_rev(nm);
        for (int i = 0; i < nm; i++) A_rev[i] = B[nm - 1 - i];
        vector<long long> H = convolution(A_rev, C); // length 2*nm-1
        vector<long long> D(nm);
        for (int k = 0; k < nm; k++) {
            // D[k] = H[nm-1-k]
            int idx = nm - 1 - k;
            if (idx < H.size()) D[k] = H[idx];
            else D[k] = 0;
        }
        
        // precompute combinations for n and m using fact and invfact
        // comb(n, i) = fact[n] * invfact[i] * invfact[n-i]
        vector<long long> comb_n(n + 1), comb_m(m + 1);
        comb_n[0] = 1;
        for (int i = 1; i <= n; i++) {
            comb_n[i] = fact[n] * invfact[i] % MOD * invfact[n - i] % MOD;
        }
        comb_m[0] = 1;
        for (int j = 1; j <= m; j++) {
            comb_m[j] = fact[m] * invfact[j] % MOD * invfact[m - j] % MOD;
        }
        
        // w[T]
        vector<long long> w(nm + 1, 0);
        for (int i = 1; i <= n; i++) {
            long long sign_i = (i % 2 == 0) ? 1 : MOD - 1;
            long long cn = comb_n[i];
            for (int j = 1; j <= m; j++) {
                int T = i * j;
                if (T > nm) continue;
                
                long long sign_j = (j % 2 == 0) ? 1 : MOD - 1;
                long long cm = comb_m[j];
                
                long long coeff = sign_i * sign_j % MOD * cn % MOD * cm % MOD;
                w[T] = (w[T] + coeff) % MOD;
            }
        }
        
        // answer
        long long ans = 0;
        for (int T = 1; T <= nm; T++) {
            long long term = w[T] * fact[T] % MOD * D[nm - T] % MOD;
            ans = (ans + term) % MOD;
        }
        if ((n + m) % 2 == 1) ans = (MOD - ans) % MOD;
        printf("%lld\n", ans);
    }
    return 0;
}
