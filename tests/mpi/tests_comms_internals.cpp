#include <fstream>
#include <array>
#include <random>
#include <exception>
#include <iostream>
#include <sstream>
#include <tuple>
#include <cstring>
#include <thread>
#include <gtest/gtest.h>
#include <mpi.h>

#include "tasktorrent/tasktorrent.hpp"

using namespace std;
using namespace ttor;

/**
 * Run with (for normal tests)
 * mpirun -n 2 ./tests_comms_internals 0 --gtest_repeat=10000 --gtest_break_on_failure --gtest_filter=-*large*
 * Or (for "large messages" tests)
 * mpirun -n 2 ./tests_comms_internals 0 --gtest_repeat=10000 --gtest_break_on_failure --gtest_filter=*large*
 */

int VERB = 0;

TEST(ttor, activeMessages)
{
    int rank = comm_rank();
    Communicator comm(MPI_COMM_WORLD, VERB);

    double local = 0.0;
    bool done = false;
    vector<double> payload = {3.14, 2.71, 9.99};
    double sum = 0;
    for (auto v_ : payload)
        sum += v_;
    auto fun1 = [&](view<double> &v) {
        for (auto vv : v)
        {
            local += vv;
        }
    };
    std::function<void()> fun2 = [&]() { done = true; };
    auto am1 = comm.make_active_msg(fun1);
    auto am2 = comm.make_active_msg(fun2);
    if (rank == 0)
    {
        auto v = view<double>(payload.data(), payload.size());
        for (int dest = 1; dest < comm_size(); dest++)
        {
            am1->blocking_send(dest, v);
            am2->blocking_send(dest);
        }
    }
    else
    {
        while (!done)
        {
            comm.progress();
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
    MPI_Barrier(MPI_COMM_WORLD);
}

struct local_t
{
    int data;
    double value;
};

TEST(ttor, blocking)
{
    int rank = comm_rank();

    local_t l = {0, 0.0};
    vector<double> payload = {3.14, 2.71, 9.99};
    Communicator comm(MPI_COMM_WORLD, VERB);
    // Define the active messages
    auto am0 = comm.make_active_msg([&](int &i, int &j) {
        l.data += i;
        l.data += j;
    });
    auto am1 = comm.make_active_msg([&](view<double> &payload) {
        for (auto v : payload)
        {
            l.value += v;
        }
    });
    auto am2 = comm.make_active_msg([&](view<const char> &message) {
        string str(message.begin(), message.end());
    });
    auto am3 = comm.make_active_msg([&]() {
    });
    // SENDER
    if (rank > 0)
    {
        int dest = 0;
        {
            int rank10 = 10 * rank;
            am0->blocking_send(dest, rank, rank10);
        }
        {
            auto v1 = view<double>(payload.data(), payload.size());
            am1->blocking_send(dest, v1);
        }
        {
            string salut = "Salut!";
            auto v2 = view<const char>(salut.data(), salut.size());
            am2->blocking_send(dest, v2);
        }
        {
            am3->blocking_send(dest);
        }
    }
    else
    {
        for (int step = 0; step < 4 * (comm_size() - 1); step++)
        {
            comm.recv_process();
        }
    }

    if (rank == 0)
    {
        int data = 0;
        for (int r = 1; r < comm_size(); r++)
        {
            data += r;
            data += 10 * r;
        }
        ASSERT_EQ(l.data, data);
        double value = 0.0;
        for (int r = 1; r < comm_size(); r++)
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
    MPI_Barrier(MPI_COMM_WORLD);
}

struct local2_t
{
    int count;
    int value;
};

TEST(ttor, nonblocking)
{
    Logger log(1000000);
    int nlpcs = comm_size() - 1;
    int expected = 0;
    for (int k = 1; k < comm_size(); k++)
    {
        expected += k;
    }
    local2_t l = {0, 0};
    Communicator comm(MPI_COMM_WORLD, VERB);
    comm.set_logger(&log);
    auto am = comm.make_active_msg([&](int &value) {
        l.value += value;
        l.count++;
    });
    if (comm_rank() > 0)
    {
        int payload = comm_rank();
        am->send(0, payload);
    }
    if (comm_rank() == 0)
    {
        while (l.count != nlpcs)
        {
            comm.progress();
        }
    }
    else
    {
        while (!comm.is_done())
        {
            comm.progress();
        }
    }
    if (comm_rank() == 0)
    {
        ASSERT_EQ(l.count, nlpcs);
        ASSERT_EQ(l.value, expected);
    }
    else
    {
        ASSERT_EQ(l.count, 0);
        ASSERT_EQ(l.value, 0);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    std::ofstream logfile;
    logfile.open("nonblocking.log." + to_string(comm_rank()));
    logfile << log;
    logfile.close();
}

TEST(mpi, many1)
{
    const int nrpcs = 100; // number of messages
    const int size = 1000; // Number of int in each message
    const int expected = nrpcs * (comm_size() - 1);
    int rank = comm_rank();
    vector<int> tosend(size, 0);
    vector<vector<int>> torecv(expected, vector<int>(size, 0));
    vector<MPI_Request> rqst_send(expected);
    vector<MPI_Request> rqst_recv(expected);
    int k = 0;
    for (int i = 0; i < nrpcs; i++)
    {
        for (int dest = 0; dest < comm_size(); dest++)
        {
            if (dest != rank)
            {
                assert(k >= 0 && k < expected);
                MPI_Isend(tosend.data(), size, MPI_INT, dest, 0, MPI_COMM_WORLD, &rqst_send[k]);
                MPI_Irecv(torecv[k].data(), size, MPI_INT, dest, 0, MPI_COMM_WORLD, &rqst_recv[k]);
                k++;
            }
        }
    }
    MPI_Waitall(expected, rqst_send.data(), MPI_STATUSES_IGNORE);
    MPI_Waitall(expected, rqst_recv.data(), MPI_STATUSES_IGNORE);
    MPI_Barrier(MPI_COMM_WORLD);
}

TEST(ttor, many2)
{
    const int nrpcs = 100; // number of messages
    const int size = 1000; // Number of int in each message
    int expected = nrpcs * (comm_size() - 1);
    struct local_t
    {
        int done;
    };
    local_t l = {0};
    Communicator comm(MPI_COMM_WORLD, VERB);
    int rank = comm_rank();
    vector<int> payload(size, rank);
    auto am = comm.make_active_msg(
        [&](int &from, int &expected, view<int> &p) {
            for (auto &i : p)
                EXPECT_EQ(i, from);
            EXPECT_TRUE(l.done < expected);
            l.done++;
        });

    for (int i = 0; i < nrpcs; i++)
    {
        for (int dest = 0; dest < comm_size(); dest++)
        {
            if (dest != rank)
            {
                auto v = view<int>(payload.data(), payload.size());
                am->send(dest, rank, expected, v);
            }
        }
    }

    while ( (!comm.is_done()) || (l.done != expected) )
    {
        comm.progress();
    }

    EXPECT_EQ(comm.get_n_msg_processed(), expected);
    EXPECT_EQ(comm.get_n_msg_queued(), expected);
    EXPECT_EQ(l.done, expected);
    
    MPI_Barrier(MPI_COMM_WORLD);
}

// Test for a potential bug in mpich with MPI_Probe and MPI_Count not returning correct values for large messages
// This need very large messages to trigger the bug
TEST(mpi, largeProbeGetCount)
{
    MPI_Datatype MPI_MEGABYTE;
    int mega = 1 << 20;
    MPI_Type_contiguous(mega, MPI_BYTE, &MPI_MEGABYTE);
    MPI_Type_commit(&MPI_MEGABYTE);

    std::vector<int> counts = {1, 5, 100, 5000};
    for(auto count: counts) {
        size_t size = static_cast<size_t>(mega) * static_cast<size_t>(count);
        char* sendbuff = (char*)calloc(size, sizeof(char));
        char* recvbuff = (char*)calloc(size, sizeof(char));
        sendbuff[0] = '1';
        sendbuff[size-1] = '7';
        MPI_Request send, recv;
        int after = (comm_rank() + 1) % (comm_size());
        int before = (comm_rank() > 0) ? (comm_rank() - 1) : (comm_size() - 1);
        MPI_Isend(sendbuff, count, MPI_MEGABYTE, after, 0, MPI_COMM_WORLD, &send);
        {
            MPI_Status status;
            MPI_Probe(before, 0, MPI_COMM_WORLD, &status);
            int probe_count = 0;
            MPI_Get_count(&status, MPI_MEGABYTE, &probe_count);
            EXPECT_EQ(probe_count, count);
        }
        MPI_Irecv(recvbuff, count, MPI_MEGABYTE, before, 0, MPI_COMM_WORLD, &recv);
        MPI_Wait(&send, MPI_STATUS_IGNORE);
        MPI_Wait(&recv, MPI_STATUS_IGNORE);
        {
            EXPECT_EQ(recvbuff[0],'1');
            EXPECT_EQ(recvbuff[size-1],'7');
        }
        
        MPI_Barrier(MPI_COMM_WORLD);
        free(sendbuff);
        free(recvbuff);
    }
}

class BreakSize : public ::testing::Test, public ::testing::WithParamInterface<tuple<double>> {};

TEST_P(BreakSize, Check) {
    double s = 0;
    std::tie(s) = GetParam();
    const size_t break_size = (1 << 22); // Larger than 1MB but smaller than 2^31
    if(VERB) printf("Size factor = %e =======================\n", s);
    Communicator comm(MPI_COMM_WORLD, VERB, break_size);
    int done = 0;
    int expected = 1;
    size_t size = break_size * s;
    char* buffer = (char*)calloc(size, sizeof(char));
    buffer[0] = 'l';
    buffer[1] = 'e';
    buffer[size/4] = 'o';
    buffer[size/2] = 'p';
    buffer[3*size/4] = 'o';
    buffer[size-2] = 'l';
    buffer[size-1] = 'd';
    auto am = comm.make_active_msg(
        [&](view<char> &p) {
            char* buffer = p.data();
            size_t actual_size = p.size();
            EXPECT_EQ(size, actual_size);
            EXPECT_EQ(buffer[0],'l');
            EXPECT_EQ(buffer[1],'e');
            EXPECT_EQ(buffer[actual_size/4],'o');
            EXPECT_EQ(buffer[actual_size/2],'p');
            EXPECT_EQ(buffer[3*actual_size/4],'o');
            EXPECT_EQ(buffer[actual_size-2],'l');
            EXPECT_EQ(buffer[actual_size-1],'d');
            done++;
        });

    auto v = view<char>(buffer, size);
    am->send( (comm_rank() + 1) % (comm_size()) , v);

    while ( (!comm.is_done()) || (done != expected) ) {
        comm.progress();
    }
    
    EXPECT_EQ(done, expected);
    MPI_Barrier(MPI_COMM_WORLD);
    free(buffer);
}

INSTANTIATE_TEST_SUITE_P(
    Ttor, BreakSize,
    ::testing::Combine(
        ::testing::Values(0.001, 0.01, 0.1, 0.5, 0.9, 1.1, 1.2, 1.3, 1.5, 2.0, 4.0, 5.5, 6.0, 7.8)
    )
);

int main(int argc, char **argv)
{
    MPI_Init(NULL, NULL);
    ::testing::InitGoogleTest(&argc, argv);
    if (argc > 1)
    {
        VERB = atoi(argv[1]);
    }
    const int return_flag = RUN_ALL_TESTS();
    MPI_Finalize();
    return return_flag;
}
