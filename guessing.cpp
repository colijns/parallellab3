#include "PCFG.h"
#include <algorithm>
#include <pthread.h>
#include <vector>
#include <string>
using namespace std;

// GEN_MODE:
// 0 = 串行 Generate，写入 q.guesses
// 1 = OpenMP Generate，写入 q.local_guesses
// 2 = pthread Generate，写入 q.local_guesses
#ifndef GEN_MODE
#define GEN_MODE 0
#endif

#ifndef GEN_THREADS
#define GEN_THREADS 8
#endif

#ifndef GEN_THRESHOLD
#define GEN_THRESHOLD 2500
#endif

void PriorityQueue::EnsureLocalBuckets(int n)
{
    if (n <= 0) n = 1;
    if ((int)local_guesses.size() != n)
    {
        local_guesses.clear();
        local_guesses.resize(n);
    }
}

void PriorityQueue::ClearGuesses()
{
    guesses.clear();
    for (auto &v : local_guesses)
    {
        v.clear();
    }
    total_guesses = 0;
}

size_t PriorityQueue::BufferedGuessCount() const
{
#if GEN_MODE == 0
    return guesses.size();
#else
    size_t total = 0;
    for (const auto &v : local_guesses)
    {
        total += v.size();
    }
    return total;
#endif
}

static segment *FindSegmentInModel(model &m, const segment &seg)
{
    if (seg.type == 1) return &m.letters[m.FindLetter(seg)];
    if (seg.type == 2) return &m.digits[m.FindDigit(seg)];
    if (seg.type == 3) return &m.symbols[m.FindSymbol(seg)];
    return nullptr;
}

static segment *PrepareGenerateTask(model &m, const PT &pt, string &prefix)
{
    prefix.clear();
    if (pt.content.empty()) return nullptr;

    if (pt.content.size() == 1)
    {
        return FindSegmentInModel(m, pt.content[0]);
    }

    int seg_idx = 0;
    for (int idx : pt.curr_indices)
    {
        if (seg_idx >= (int)pt.content.size() - 1) break;

        segment *seg = FindSegmentInModel(m, pt.content[seg_idx]);
        if (seg != nullptr)
        {
            prefix += seg->ordered_values[idx];
        }
        seg_idx++;
    }

    int last = (int)pt.content.size() - 1;
    return FindSegmentInModel(m, pt.content[last]);
}

void PriorityQueue::CalProb(PT &pt)
{
    pt.prob = pt.preterm_prob;
    int index = 0;

    for (int idx : pt.curr_indices)
    {
        if (pt.content[index].type == 1)
        {
            pt.prob *= m.letters[m.FindLetter(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.letters[m.FindLetter(pt.content[index])].total_freq;
        }
        if (pt.content[index].type == 2)
        {
            pt.prob *= m.digits[m.FindDigit(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.digits[m.FindDigit(pt.content[index])].total_freq;
        }
        if (pt.content[index].type == 3)
        {
            pt.prob *= m.symbols[m.FindSymbol(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.symbols[m.FindSymbol(pt.content[index])].total_freq;
        }
        index += 1;
    }
}

void PriorityQueue::init()
{
    for (PT pt : m.ordered_pts)
    {
        for (segment seg : pt.content)
        {
            if (seg.type == 1)
            {
                pt.max_indices.emplace_back(m.letters[m.FindLetter(seg)].ordered_values.size());
            }
            if (seg.type == 2)
            {
                pt.max_indices.emplace_back(m.digits[m.FindDigit(seg)].ordered_values.size());
            }
            if (seg.type == 3)
            {
                pt.max_indices.emplace_back(m.symbols[m.FindSymbol(seg)].ordered_values.size());
            }
        }
        pt.preterm_prob = float(m.preterm_freq[m.FindPT(pt)]) / m.total_preterm;
        CalProb(pt);
        priority.emplace_back(pt);
    }
}

void PriorityQueue::PopNext()
{
#if GEN_MODE == 1
    Generate_openmp(priority.front());
#elif GEN_MODE == 2
    Generate_pthread(priority.front());
#else
    Generate(priority.front());
#endif

    vector<PT> new_pts = priority.front().NewPTs();
    for (PT pt : new_pts)
    {
        CalProb(pt);
        for (auto iter = priority.begin(); iter != priority.end(); iter++)
        {
            if (iter != priority.end() - 1 && iter != priority.begin())
            {
                if (pt.prob <= iter->prob && pt.prob > (iter + 1)->prob)
                {
                    priority.emplace(iter + 1, pt);
                    break;
                }
            }
            if (iter == priority.end() - 1)
            {
                priority.emplace_back(pt);
                break;
            }
            if (iter == priority.begin() && iter->prob < pt.prob)
            {
                priority.emplace(iter, pt);
                break;
            }
        }
    }

    priority.erase(priority.begin());
}

vector<PT> PT::NewPTs()
{
    vector<PT> res;

    if (content.size() == 1)
    {
        return res;
    }
    else
    {
        int init_pivot = pivot;
        for (int i = pivot; i < (int)curr_indices.size() - 1; i += 1)
        {
            curr_indices[i] += 1;
            if (curr_indices[i] < max_indices[i])
            {
                pivot = i;
                res.emplace_back(*this);
            }
            curr_indices[i] -= 1;
        }
        pivot = init_pivot;
        return res;
    }

    return res;
}

// 串行版本：保留原逻辑，结果写入 guesses
void PriorityQueue::Generate(PT pt)
{
    CalProb(pt);

    string prefix;
    segment *a = PrepareGenerateTask(m, pt, prefix);
    if (a == nullptr) return;

    int total = (int)a->ordered_values.size();
    if (total <= 0) return;

    guesses.reserve(guesses.size() + total);
    for (int i = 0; i < total; i++)
    {
        guesses.emplace_back(prefix + a->ordered_values[i]);
    }
    total_guesses += total;
}

// OpenMP 版本：学长式结构，结果保留在线程局部数组，不再合并到 guesses
void PriorityQueue::Generate_openmp(PT pt)
{
    CalProb(pt);

    string prefix;
    segment *a = PrepareGenerateTask(m, pt, prefix);
    if (a == nullptr) return;

    int total = (int)a->ordered_values.size();
    if (total <= 0) return;

    EnsureLocalBuckets(GEN_THREADS);

    // 小任务直接串行，但仍分散写入 local_guesses，避免后续再合并
    if (total < GEN_THRESHOLD)
    {
        for (int t = 0; t < GEN_THREADS; t++)
        {
            int start = t * total / GEN_THREADS;
            int end = (t + 1) * total / GEN_THREADS;
            local_guesses[t].reserve(local_guesses[t].size() + (end - start));
        }

        for (int i = 0; i < total; i++)
        {
            int t = i % GEN_THREADS;
            local_guesses[t].emplace_back(prefix + a->ordered_values[i]);
        }

        total_guesses += total;
        return;
    }

    for (int t = 0; t < GEN_THREADS; t++)
    {
        int start = t * total / GEN_THREADS;
        int end = (t + 1) * total / GEN_THREADS;
        local_guesses[t].reserve(local_guesses[t].size() + (end - start));
    }

#ifdef _OPENMP
#pragma omp parallel num_threads(GEN_THREADS)
#endif
    {
#ifdef _OPENMP
        int tid = omp_get_thread_num();
        int nth = omp_get_num_threads();
#else
        int tid = 0;
        int nth = 1;
#endif
        int start = tid * total / nth;
        int end = (tid + 1) * total / nth;

        vector<string> &local = local_guesses[tid];
        for (int i = start; i < end; i++)
        {
            local.emplace_back(prefix + a->ordered_values[i]);
        }
    }

    total_guesses += total;
}

struct PthreadBucketArgs
{
    int start;
    int end;
    string prefix;
    vector<string> *values;
    vector<string> *local;
};

static void *PthreadBucketWorker(void *arg)
{
    PthreadBucketArgs *args = (PthreadBucketArgs *)arg;
    vector<string> &local = *(args->local);

    for (int i = args->start; i < args->end; i++)
    {
        local.emplace_back(args->prefix + (*(args->values))[i]);
    }

    return nullptr;
}

// pthread 版本：动态线程，但不合并 guesses，哈希阶段直接读 local_guesses
void PriorityQueue::Generate_pthread(PT pt)
{
    CalProb(pt);

    string prefix;
    segment *a = PrepareGenerateTask(m, pt, prefix);
    if (a == nullptr) return;

    int total = (int)a->ordered_values.size();
    if (total <= 0) return;

    EnsureLocalBuckets(GEN_THREADS);

    if (total < GEN_THRESHOLD)
    {
        for (int t = 0; t < GEN_THREADS; t++)
        {
            int start = t * total / GEN_THREADS;
            int end = (t + 1) * total / GEN_THREADS;
            local_guesses[t].reserve(local_guesses[t].size() + (end - start));
        }

        for (int i = 0; i < total; i++)
        {
            int t = i % GEN_THREADS;
            local_guesses[t].emplace_back(prefix + a->ordered_values[i]);
        }

        total_guesses += total;
        return;
    }

    pthread_t threads[GEN_THREADS];
    PthreadBucketArgs args[GEN_THREADS];

    for (int t = 0; t < GEN_THREADS; t++)
    {
        int start = t * total / GEN_THREADS;
        int end = (t + 1) * total / GEN_THREADS;

        local_guesses[t].reserve(local_guesses[t].size() + (end - start));

        args[t].start = start;
        args[t].end = end;
        args[t].prefix = prefix;
        args[t].values = &a->ordered_values;
        args[t].local = &local_guesses[t];

        pthread_create(&threads[t], nullptr, PthreadBucketWorker, &args[t]);
    }

    for (int t = 0; t < GEN_THREADS; t++)
    {
        pthread_join(threads[t], nullptr);
    }

    total_guesses += total;
}
