#include "PCFG.h"
#include <chrono>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <array>
#include <vector>
#include "md5.h"
using namespace std;
using namespace chrono;

#ifndef GEN_MODE
#define GEN_MODE 0
#endif

#ifndef GEN_THREADS
#define GEN_THREADS 8
#endif

#ifndef GENERATE_LIMIT
#define GENERATE_LIMIT 10000000
#endif

#ifndef USE_SIMD_HASH
#define USE_SIMD_HASH 1
#endif

#if USE_SIMD_HASH
// 与当前 md5.cpp 中的 SIMD 批量哈希接口保持一致：
// results == nullptr 时，只进行性能测试，不保存每个口令的 MD5 结果。
void MD5HashBatchSIMD(
    const vector<string>& passwords,
    vector<array<bit32, 4>>* results
);
#endif

static void HashVector(const vector<string> &v)
{
    if (v.empty()) return;

#if USE_SIMD_HASH
    MD5HashBatchSIMD(v, nullptr);
#else
    bit32 state[4];
    for (const string &pw : v)
    {
        MD5Hash(pw, state);
    }
#endif
}

static void HashBufferedGuesses(PriorityQueue &q)
{
#if GEN_MODE == 0
    // 串行版本：结果保存在 q.guesses 中
    HashVector(q.guesses);
#else
    // 多线程版本：结果按线程保存在 q.local_guesses 中，避免合并到 q.guesses
    for (const auto &bucket : q.local_guesses)
    {
        HashVector(bucket);
    }
#endif
}

int main()
{
    cout << "Testing MD5Hash correctness..." << endl;
    string test_pws[8] = {"123456", "password", "12345678", "qwerty", "123456789", "12345", "1234", "111111"};
    string test_hashes[8] = {
        "e10adc3949ba59abbe56e057f20f883e",
        "5f4dcc3b5aa765d61d8327deb882cf99",
        "25d55ad283aa400af464c76d713c07ad",
        "d8578edf8458ce06fbc5bb76a58c5ca4",
        "25f9e794323b453885f5181f1b624d0b",
        "827ccb0eea8a706c4c34a16891f84e7b",
        "81dc9bdb52d04dc20036dbd8313ed055",
        "96e79218965eb72c92a549dd5a330112"
    };

    for (int i = 0; i < 8; i++)
    {
        bit32 state[4];
        MD5Hash(test_pws[i], state);
        stringstream ss;
        for (int i1 = 0; i1 < 4; i1 += 1)
        {
            ss << setw(8) << setfill('0') << hex << state[i1];
        }
        if (ss.str() != test_hashes[i])
        {
            cout << "MD5Hash test failed for " << test_pws[i] << "!" << endl;
            cout << "Expected: " << test_hashes[i] << "\nGot:      " << ss.str() << endl;
            return 1;
        }
    }
    cout << "MD5Hash test passed!" << endl;

    double time_hash = 0;
    double time_guess = 0;
    double time_train = 0;

    PriorityQueue q;

    auto start_train = system_clock::now();
    q.m.train("/guessdata/Rockyou-singleLined-full.txt");
    q.m.order();
    auto end_train = system_clock::now();
    auto duration_train = duration_cast<microseconds>(end_train - start_train);
    time_train = double(duration_train.count()) * microseconds::period::num / microseconds::period::den;

    q.init();
    cout << "here" << endl;

    int curr_num = 0;
    int history = 0;
    auto start = system_clock::now();

    while (!q.priority.empty())
    {
        q.PopNext();

        if (q.total_guesses - curr_num >= 100000)
        {
            cout << "Guesses generated: " << history + q.total_guesses << endl;
            curr_num = q.total_guesses;

            if (history + q.total_guesses > GENERATE_LIMIT)
            {
                auto end = system_clock::now();
                auto duration = duration_cast<microseconds>(end - start);
                time_guess = double(duration.count()) * microseconds::period::num / microseconds::period::den;
                cout << "Guess time:" << time_guess - time_hash << "seconds" << endl;
                cout << "Hash time:" << time_hash << "seconds" << endl;
                cout << "Train time:" << time_train << "seconds" << endl;
                break;
            }
        }

        if (curr_num > 1000000)
        {
            auto start_hash = system_clock::now();

            HashBufferedGuesses(q);

            auto end_hash = system_clock::now();
            auto duration = duration_cast<microseconds>(end_hash - start_hash);
            time_hash += double(duration.count()) * microseconds::period::num / microseconds::period::den;

            history += curr_num;
            curr_num = 0;
            q.ClearGuesses();
        }
    }

    return 0;
}
