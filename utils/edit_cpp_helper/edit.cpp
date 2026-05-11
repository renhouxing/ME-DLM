#include <vector>
#include <random>
#include <utility>
#include <string>
#include <algorithm>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

struct Op {
    enum Type { KEEP, REPLACE, DELETE_, INSERT } type;
    int a = 0;
    int b = 0;
};

struct LabelResult {
    std::vector<int> token_target;
    std::vector<int> next_target;
};

LabelResult compute_labels_cpp(
    const std::vector<int>& noisy,
    const std::vector<int>& clean,
    int del_id = -1,
    int eos_id = -2,
    int ignore_id = -100,
    bool pred_clean_token = false
) {
    int n = (int)noisy.size();
    int m = (int)clean.size();

    std::vector<std::vector<int>> dp(n + 1, std::vector<int>(m + 1, 0));
    for (int i = 1; i <= n; ++i) dp[i][0] = i;
    for (int j = 1; j <= m; ++j) dp[0][j] = j;

    for (int i = 1; i <= n; ++i) {
        for (int j = 1; j <= m; ++j) {
            if (noisy[i - 1] == clean[j - 1]) dp[i][j] = dp[i - 1][j - 1];
            else dp[i][j] = dp[i - 1][j - 1] + 1;                 // replace
            dp[i][j] = std::min(dp[i][j], dp[i - 1][j] + 1);      // delete
            dp[i][j] = std::min(dp[i][j], dp[i][j - 1] + 1);      // insert
        }
    }

    std::vector<Op> ops;
    int i = n, j = m;
    while (i > 0 || j > 0) {
        if (i > 0 && j > 0 && noisy[i - 1] == clean[j - 1] && dp[i][j] == dp[i - 1][j - 1]) {
            ops.push_back(Op{Op::KEEP, noisy[i - 1], 0});
            --i; --j;
        } else if (i > 0 && j > 0 && dp[i][j] == dp[i - 1][j - 1] + 1) {
            ops.push_back(Op{Op::REPLACE, noisy[i - 1], clean[j - 1]});
            --i; --j;
        } else if (i > 0 && dp[i][j] == dp[i - 1][j] + 1) {
            ops.push_back(Op{Op::DELETE_, noisy[i - 1], 0});
            --i;
        } else {
            ops.push_back(Op{Op::INSERT, 0, clean[j - 1]});
            --j;
        }
    }
    std::reverse(ops.begin(), ops.end());

    std::vector<int> token_target;
    if (pred_clean_token) token_target = noisy;
    else token_target.assign(n, ignore_id);

    std::vector<std::vector<int>> gap_insert(n + 1);

    int idx_noisy = 0;
    int g = 0;
    for (const auto& op : ops) {
        if (op.type == Op::INSERT) {
            gap_insert[g].push_back(op.b);
        } else if (op.type == Op::KEEP) {
            idx_noisy += 1;
            g += 1;
        } else if (op.type == Op::REPLACE) {
            token_target[idx_noisy] = op.b;
            idx_noisy += 1;
            g += 1;
        } else if (op.type == Op::DELETE_) {
            token_target[idx_noisy] = del_id;
            idx_noisy += 1;
            g += 1;
        }
    }

    std::vector<int> next_target(n + 1, ignore_id);
    for (int k = 0; k < n + 1; ++k) {
        if (!gap_insert[k].empty()) next_target[k] = gap_insert[k][0];
    }

    for (int k = 0; k < (int)token_target.size(); ++k) {
        if (k < (int)token_target.size() - 1) {
            int a = token_target[k + 1];
            int b = next_target[k + 1];
            if (a == b && b != ignore_id) {
                if (k + 2 < (int)next_target.size()) {
                    std::swap(next_target[k + 1], next_target[k + 2]);
                }
            }
        }

        if (next_target[k] == ignore_id && (k == 0 || token_target[k - 1] != del_id)) {
            int target = eos_id;
            int jj = k;
            while (jj < (int)token_target.size() && token_target[jj] == del_id) {
                jj += 1;
            }
            if (jj < (int)token_target.size()) {
                target = token_target[jj];
            }
            next_target[k] = target;
        }
    }

    if (!next_target.empty() && next_target.back() == ignore_id) {
        next_target.back() = eos_id;
    }

    return {token_target, next_target};
}

PYBIND11_MODULE(edit_cpp, m) {
    m.def(
        "compute_labels_cpp",
        [](
            const std::vector<int>& noisy,
            const std::vector<int>& clean,
            int del_id,
            int eos_id,
            int ignore_id,
            bool pred_clean_token
        ) {
            LabelResult r = compute_labels_cpp(noisy, clean, del_id, eos_id, ignore_id, pred_clean_token);
            return pybind11::make_tuple(r.token_target, r.next_target);
        },
        pybind11::arg("noisy"),
        pybind11::arg("clean"),
        pybind11::arg("del_id") = -1,
        pybind11::arg("eos_id") = -2,
        pybind11::arg("ignore_id") = -100,
        pybind11::arg("pred_clean_token") = false
    );
}