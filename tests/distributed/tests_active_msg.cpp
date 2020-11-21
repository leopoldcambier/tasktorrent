#include <fstream>
#include <array>
#include <random>
#include <exception>
#include <iostream>
#include <sstream>
#include <cstring>

#include <gtest/gtest.h>

#include "tasktorrent/tasktorrent.hpp"

using namespace std;
using namespace ttor;

int VERB = 0;

TEST(Communicator, SendToSelf)
{
    int rank = comms_world_rank();
    auto comm = make_communicator_world(VERB);

    bool done = false;
    auto fun = [&]() {
        done = true;
    };
    auto am = comm->make_active_msg(fun);
    am->send(rank);
    {
        while (!done)
        {
            comm->progress();
        }
    }
    ASSERT_TRUE(done);
    comms_world_barrier();
}

TEST(Communicator, ActiveMessages)
{
    int rank = comms_world_rank();
    int nranks = comms_world_size();
    auto comm = make_communicator_world(VERB);
    if(nranks < 2) {
        printf("Need at least 2 ranks\n");
        return;
    }

    double local = 0.0;
    bool done = false;
    vector<double> payload = {3.14, 2.71, 9.99};
    double sum = 0;
    for (auto v_ : payload)
        sum += v_;
    auto fun1 = [&](const view<double> &v) {
        for (auto vv : v)
        {
            local += vv;
        }
    };
    std::function<void()> fun2 = [&]() { done = true; };
    auto am1 = comm->make_active_msg(fun1);
    auto am2 = comm->make_active_msg(fun2);
    if (rank == 0)
    {
        auto v = make_view(payload.data(), payload.size());
        for (int dest = 1; dest < comms_world_size(); dest++)
        {
            am1->blocking_send(dest, v);
            am2->blocking_send(dest);
        }
    }
    else
    {
        while (!done)
        {
            comm->progress();
        }
    }
    if (rank == 0)
    {
        ASSERT_FALSE(done);
        ASSERT_EQ(local, 0.0);
    }
    else
    {
        ASSERT_TRUE(done);
        ASSERT_EQ(local, sum);
    }
    comms_world_barrier();
}

/** Test the various ways to build and call an AM **/
TEST(Communicator, ActiveMessages_Value_ConstRef)
{
    int rank = comms_world_rank();
    int nranks = comms_world_size();
    if(nranks < 2) {
        printf("Need at least 2 ranks\n");
        return;
    }
    auto comm = make_communicator_world(VERB);
    int rcvd = 0;
    int expt = 16;

    int one = 1;
    int& one_ref = one;
    const int& one_const_ref = one;

    std::vector<int> data(10, -1);
    const std::vector<int> data_const(10, -1);

    auto am1 = comm->make_active_msg([&](int){ rcvd++; });
    auto am2 = comm->make_active_msg([&](const int&){ rcvd++; });

    auto vv1 = make_view(data.data(), data.data() + data.size());
    auto vv2 = make_view(data_const.data(), data_const.data() + data_const.size());

    auto am3 = comm->make_active_msg([&](const view<int>&){ rcvd++; });
    auto am4 = comm->make_active_msg([&](view<int>){ rcvd++; });

    if (rank == 0)
    {
        for (int dest = 1; dest < comms_world_size(); dest++)
        {
            am1->blocking_send(dest, 1);
            am1->blocking_send(dest, one);
            am1->blocking_send(dest, one_ref);
            am1->blocking_send(dest, one_const_ref);

            am2->blocking_send(dest, 1);
            am2->blocking_send(dest, one);
            am2->blocking_send(dest, one_ref);
            am2->blocking_send(dest, one_const_ref);

            am3->blocking_send(dest, vv1);
            am3->blocking_send(dest, vv2);
            am3->blocking_send(dest, make_view(data.data(), data.data() + data.size()));
            am3->blocking_send(dest, make_view(data_const.data(), data_const.data() + data_const.size()));

            am4->blocking_send(dest, vv1);
            am4->blocking_send(dest, vv2);
            am4->blocking_send(dest, make_view(data.data(), data.data() + data.size()));
            am4->blocking_send(dest, make_view(data_const.data(), data_const.data() + data_const.size()));
        }
    }
    else
    {
        while (expt != rcvd)
        {
            comm->progress();
        }
    }
    comms_world_barrier();
}

struct local_t
{
    int data;
    double value;
};

TEST(ActiveMsg, Blocking)
{
    const int rank = comms_world_rank();
    const int nranks = comms_world_size();
    if(nranks < 2) {
        printf("Need at least 2 ranks\n");
        return;
    }

    local_t l = {0, 0.0};
    vector<double> payload = {3.14, 2.71, 9.99};
    auto comm = make_communicator_world(VERB);
    // Define the active messages
    int msg_rcvd = 0;
    auto am0 = comm->make_active_msg([&](const int &i, const int &j) {
        l.data += i;
        l.data += j;
        msg_rcvd++;
    });
    auto am1 = comm->make_active_msg([&](const view<double> &payload) {
        for (auto v : payload)
        {
            l.value += v;
        }
        msg_rcvd++;
    });
    auto am2 = comm->make_active_msg([&](const view<char> &message) {
        string str(message.begin(), message.end());
        msg_rcvd++;
    });
    auto am3 = comm->make_active_msg([&]() {
        msg_rcvd++;
    });
    // SENDER
    if (rank > 0)
    {
        int dest = 0;
        {
            int rank10 = 10 * rank;
            int from = rank;
            am0->blocking_send(dest, from, rank10);
        }
        {
            auto v1 = make_view(payload.data(), payload.size());
            am1->blocking_send(dest, v1);
        }
        {
            string salut("Salut!");
            auto v2 = make_view(salut.data(), salut.size());
            am2->blocking_send(dest, v2);
        }
        {
            am3->blocking_send(dest);
        }
    }
    else
    {
        while(msg_rcvd != 4 * (nranks - 1)) {
            comm->progress();
        }
    }

    if (rank == 0)
    {
        int data = 0;
        for (int r = 1; r < nranks; r++)
        {
            data += r;
            data += 10 * r;
        }
        ASSERT_EQ(l.data, data);
        double value = 0.0;
        for (int r = 1; r < nranks; r++)
        {
            for (auto v : payload)
            {
                value += v;
            }
        }
        ASSERT_EQ(l.value, value);
    }
    else
    {
        ASSERT_EQ(l.value, 0.0);
        ASSERT_EQ(l.data, 0);
    }
    ASSERT_EQ(msg_rcvd, rank == 0 ? 4 * (nranks - 1) : 0);
    ASSERT_EQ(rank > 0 ? 4 : 0, comm->get_n_msg_queued()) << rank << " " << nranks;
    ASSERT_EQ(msg_rcvd, comm->get_n_msg_processed()) << rank << " " << nranks;
    comms_world_barrier();
}

struct local2_t
{
    int count;
    int value;
};

TEST(ActiveMsg, Nonblocking)
{
    Logger log(1000000);
    int nlpcs = comms_world_size() - 1;
    int expected = 0;
    for (int k = 1; k < comms_world_size(); k++)
    {
        expected += k;
    }
    local2_t l = {0, 0};
    auto comm = make_communicator_world(VERB);
    comm->set_logger(&log);
    auto am = comm->make_active_msg([&](const int &value) {
        l.value += value;
        l.count++;
    });
    if (comms_world_rank() > 0)
    {
        int payload = comms_world_rank();
        am->send(0, payload);
    }
    if (comms_world_rank() == 0)
    {
        while (l.count != nlpcs)
        {
            comm->progress();
        }
    }
    else
    {
        while (!comm->is_done())
        {
            comm->progress();
        }
    }
    if (comms_world_rank() == 0)
    {
        ASSERT_EQ(l.count, nlpcs);
        ASSERT_EQ(l.value, expected);
    }
    else
    {
        ASSERT_EQ(l.count, 0);
        ASSERT_EQ(l.value, 0);
    }

    comms_world_barrier();

    std::ofstream logfile;
    logfile.open("nonblocking.log." + to_string(comms_world_rank()));
    logfile << log;
    logfile.close();
}

TEST(ActiveMsgBody, TwoStepsBlocking)
{
    const int rank = comms_world_rank();
    const int nranks = comms_world_size();
    if(nranks < 2) {
        printf("Need at least 2 ranks\n");
        return;
    }
    const int N = 64;

    auto comm = make_communicator_world(VERB);

    int msg_rcvd = 0;
    vector<vector<int>> data(nranks, vector<int>(N,0));
    vector<bool> received(nranks, false);
    vector<bool> completed(nranks, false);
    received[rank] = true;
    completed[rank] = true;
    for(int i = 0; i < N; i++) data[rank][i] = rank * N + i;

    auto am = comm->make_large_active_msg(
        [&](const int& from, const int& to){
            EXPECT_EQ(to, rank);
            EXPECT_FALSE(received[from]);
            received[from] = true;
            msg_rcvd++;
        },
        [&](const int& from, const int& to){
            EXPECT_EQ(to, rank);
            int* buf = data[from].data();
            return buf;
        },
        [&](const int& from, const int& to){
            EXPECT_EQ(from, rank);
            EXPECT_FALSE(completed[to]);
            completed[to] = true;
        });

    for(int r = 0; r < nranks; r++) {
        if(r != rank) {
            int from = rank;
            auto v = make_view(data[rank].data(), data[rank].size());
            am->blocking_send_large(r, v, from, r);
        }
    }
    while(msg_rcvd != (nranks - 1)) {
        comm->progress();
        EXPECT_EQ(msg_rcvd, comm->get_n_msg_processed());
    }

    for(int r = 0; r < nranks; r++) {
        if(r != rank) {
            EXPECT_TRUE(received[r]);
            EXPECT_TRUE(completed[r]) << "Completed false at " << r << " for rank " << rank;
            for(int i = 0; i < N; i++) {
                EXPECT_EQ(data[r][i], r * N + i);
            }
        }
    }

    EXPECT_EQ(comm->get_n_msg_processed(), nranks-1);
    EXPECT_EQ(comm->get_n_msg_queued(), nranks-1);
    EXPECT_EQ(msg_rcvd, nranks-1);

    comms_world_barrier();
}


TEST(ActiveMsg, Many)
{
    const int nrpcs = 100; // number of messages
    const int size = 1000; // Number of int in each message
    int expected = nrpcs * (comms_world_size() - 1);
    struct local_t
    {
        int done;
    };
    local_t l = {0};
    auto comm = make_communicator_world(VERB);
    int rank = comms_world_rank();
    vector<int> payload(size, rank);
    auto am = comm->make_active_msg(
        [&](const int &from, const int &expected, const view<int> &p) {
            for (auto &i : p)
                EXPECT_EQ(i, from);
            EXPECT_TRUE(l.done < expected);
            l.done++;
        });

    for (int i = 0; i < nrpcs; i++)
    {
        for (int dest = 0; dest < comms_world_size(); dest++)
        {
            if (dest != rank)
            {
                auto v = make_view(payload.data(), payload.size());
                am->send(dest, rank, expected, v);
            }
        }
    }

    while ( (!comm->is_done()) || (l.done != expected) )
    {
        comm->progress();
    }

    EXPECT_EQ(comm->get_n_msg_processed(), expected);
    EXPECT_EQ(comm->get_n_msg_queued(), expected);
    EXPECT_EQ(l.done, expected);
    
    comms_world_barrier();
}

TEST(ActiveMsg, TwoStepsMixedNonBlocking)
{
    const int rank = comms_world_rank();
    const int nranks = comms_world_size();
    const int N = 64;

    auto comm = make_communicator_world(VERB);

    vector<vector<int>> data(nranks, vector<int>(N,0));
    vector<int> received(nranks, 0);
    vector<int> completed(nranks, 0);
    const int expected = 3 * nranks;
    int done = 0;
    for(int i = 0; i < N; i++) data[rank][i] = rank * N + i;

    auto am0 = comm->make_active_msg(
        [&](const int& from, const int& dest){
            EXPECT_EQ(dest, rank);
            received[from] ++;
            done++;
        });

    auto am1 = comm->make_large_active_msg(
        [&](const int& from, const int& dest){
            EXPECT_EQ(dest, rank);
            received[from] ++;
            done++;
        },
        [&](const int& from, const int& dest){
            EXPECT_EQ(dest, rank);
            int* buf = data[from].data();
            return buf;
        },
        [&](const int& from, const int& dest) {
            EXPECT_EQ(from, rank);
            EXPECT_EQ(completed[dest], 0);
            completed[dest]++;
        });

    auto am2 = comm->make_active_msg(
        [&](const int& from, const int& dest){
            EXPECT_EQ(dest, rank);
            received[from] ++;
            done++;
        });

    for(int dest = 0; dest < nranks; dest++) {
        int from = rank;
        auto v = make_view(data[from].data(), data[from].size());
        am0->send(dest, from, dest);
        am1->send_large(dest, v, from, dest);
        am2->send(dest, from, dest);
    }

    // While there is something to send or we haven't received everything
    while ( (!comm->is_done()) || (done != expected) ) {
        comm->progress();
    }

    for(int r = 0; r < nranks; r++) {
        EXPECT_EQ(completed[r], 1);
        EXPECT_EQ(received[r], 3);
        for(int i = 0; i < N; i++) {
            EXPECT_EQ(data[r][i], r * N + i);
        }
    }

    EXPECT_EQ(comm->get_n_msg_processed(), 3 * nranks);
    EXPECT_EQ(comm->get_n_msg_queued(), 3 * nranks);

    comms_world_barrier();
}

int main(int argc, char **argv)
{
    comms_init();
    ::testing::InitGoogleTest(&argc, argv);
    if (argc > 1)
    {
        VERB = atoi(argv[1]);
    }
    const int return_flag = RUN_ALL_TESTS();
    comms_finalize();
    return return_flag;
}
