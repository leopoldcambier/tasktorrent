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

int VERB = 0;

TEST(MPI, Many1)
{
    const int nrpcs = 100; // number of messages
    const int size = 1000; // Number of int in each message
    const int expected = nrpcs * (comms_world_size() - 1);
    int rank = comms_world_rank();
    vector<int> tosend(size, 0);
    vector<vector<int>> torecv(expected, vector<int>(size, 0));
    vector<MPI_Request> rqst_send(expected);
    vector<MPI_Request> rqst_recv(expected);
    int k = 0;
    for (int i = 0; i < nrpcs; i++)
    {
        for (int dest = 0; dest < comms_world_size(); dest++)
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
    comms_world_barrier();
}

// Test for a potential bug in mpich with MPI_Probe and MPI_Count not returning correct values for large messages
// This need very large messages to trigger the bug
TEST(MPI, ProbeGetCountlarge)
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
        int after = (comms_world_rank() + 1) % (comms_world_size());
        int before = (comms_world_rank() > 0) ? (comms_world_rank() - 1) : (comms_world_size() - 1);
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
        
        comms_world_barrier();
        free(sendbuff);
        free(recvbuff);
    }
}

int main(int argc, char **argv)
{
    MPI_Init(NULL, NULL);
    ::testing::InitGoogleTest(&argc, argv);
    if (argc > 1)
    {
        VERB = atoi(argv[1]);
    }
    const int return_flag = RUN_ALL_TESTS();
    comms_finalize();
    return return_flag;
}