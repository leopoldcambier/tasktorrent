#include <vector>
#include <gtest/gtest.h>
#include <mpi.h>

#include "tasktorrent/tasktorrent.hpp"

using namespace std;
using namespace ttor;

int VERB = 0;

TEST(ActiveMsgLarge, twoSteps)
{
    const int rank = comm_rank();
    const int nranks = comm_size();
    const int N = 64;

    Communicator comm(VERB);

    vector<vector<int>> data(nranks, vector<int>(N,0));
    vector<bool> received(nranks, false);
    received[rank] = true;
    for(int i = 0; i < N; i++) data[rank][i] = rank * N + i;

    auto am = comm.make_large_active_msg(
        [&](int& from){
            received[from] = true;
        },
        [&](int& from){
            char* buf = (char*)(data[from].data());
            return buf;
        });

    for(int r = 0; r < nranks; r++) {
        int from = rank;
        if(r != rank) {
            auto v = view<int>(data[rank].data(), data[rank].size());
            am->blocking_send_large(r, v, from);
        }
    }

    for(int r = 0; r < nranks; r++) {
        if(r != rank) comm.recv_process();
    }

    for(int r = 0; r < nranks; r++) {
        EXPECT_TRUE(received[r]);
        for(int i = 0; i < N; i++) {
            EXPECT_EQ(data[r][i], r * N + i);
        }
    }

    EXPECT_EQ(comm.get_n_msg_processed(), nranks-1);
    EXPECT_EQ(comm.get_n_msg_queued(), nranks-1);

    MPI_Barrier(MPI_COMM_WORLD); // recv_process doesn't filter per source, so yet we really need this
}

TEST(ActiveMsgLarge, mixed)
{
    const int rank = comm_rank();
    const int nranks = comm_size();
    const int N = 64;

    Communicator comm(VERB);

    vector<vector<int>> data(nranks, vector<int>(N,0));
    vector<int> received(nranks, 0);
    for(int i = 0; i < N; i++) data[rank][i] = rank * N + i;

    auto am0 = comm.make_active_msg(
        [&](int& from){
            received[from] ++;
        });

    auto am1 = comm.make_large_active_msg(
        [&](int& from){
            received[from] ++;
        },
        [&](int& from){
            char* buf = (char*)(data[from].data());
            return buf;
        });

    auto am2 = comm.make_active_msg(
        [&](int& from){
            received[from] ++;
        });

    for(int r = 0; r < nranks; r++) {
        int from = rank;
        if(r != rank) {
            auto v = view<int>(data[rank].data(), data[rank].size());
            am0->blocking_send(r, from);
            am1->blocking_send_large(r, v, from);
            am2->blocking_send(r, from);
        }
    }

    for(int r = 0; r < nranks; r++) {
        if(r != rank) {
            comm.recv_process();
            comm.recv_process();
            comm.recv_process();
        }
    }

    for(int r = 0; r < nranks; r++) {
        if(r != rank) {
            EXPECT_EQ(received[r], 3);
            for(int i = 0; i < N; i++) {
                EXPECT_EQ(data[r][i], r * N + i);
            }
        } else {
            EXPECT_EQ(received[r], 0);
        }
    }

    EXPECT_EQ(comm.get_n_msg_processed(), 3 * (nranks - 1));
    EXPECT_EQ(comm.get_n_msg_queued(), 3 * (nranks - 1));

    MPI_Barrier(MPI_COMM_WORLD); // recv_process doesn't filter per source, so yet we really need this

}

TEST(ActiveMsgLarge, multipleBodiesBreakSize)
{
    vector<double> sizes = {0.001, 0.1, 0.9, 1.1, 1.5, 4.0, 8.0}; 
    const size_t break_size = (1 << 22); // Larger than 1MB but smaller than 2^31
    for(auto s_header: sizes) {
        for(auto s_body: sizes) {
            if(VERB) printf("Size factor = %e, %e=======================\n", s_header, s_body);
            Communicator comm(VERB, break_size);
            int done = 0;
            int rcvd = 0;
            int expected = 1;
            size_t header_size = break_size * s_header;
            size_t body_size = break_size * s_body;
            char* header_buffer = (char*)calloc(header_size, sizeof(char));
            char* send_buffer = (char*)calloc(body_size, sizeof(char));
            char* recv_buffer = (char*)calloc(body_size, sizeof(char));
            send_buffer[0] = 'l';
            send_buffer[body_size-1] = 'd';
            header_buffer[0] = 'l';
            header_buffer[header_size-1] = 'd';
            auto am = comm.make_large_active_msg(
                [&](view<char>& header) {
                    EXPECT_EQ(done, 0);
                    EXPECT_EQ(rcvd, 1);
                    EXPECT_EQ(header.size(), header_size);
                    EXPECT_EQ(header.data()[0], 'l');
                    EXPECT_EQ(header.data()[header_size-1], 'd');
                    EXPECT_EQ(recv_buffer[0],'l');
                    EXPECT_EQ(recv_buffer[body_size-1],'d');
                    done++;
                },
                [&](view<char>& header) {
                    EXPECT_EQ(header.size(), header_size);
                    EXPECT_EQ(header.data()[0], 'l');
                    EXPECT_EQ(header.data()[header_size-1], 'd');
                    EXPECT_EQ(done, 0);
                    EXPECT_EQ(rcvd, 0);
                    rcvd++;
                    return recv_buffer;
                });

            auto body = view<char>(send_buffer, body_size);
            auto header = view<char>(header_buffer, header_size);
            int dest = (comm_rank() + 1) % (comm_size());
            am->send_large(dest, body, header);

            while ( (!comm.is_done()) || (done != expected) ) {
                comm.progress();
            }
            
            EXPECT_EQ(done, expected);
            EXPECT_EQ(rcvd, expected);
            MPI_Barrier(MPI_COMM_WORLD);
            free(send_buffer);
            free(recv_buffer);
            free(header_buffer);
        }
    }
}

int main(int argc, char **argv)
{
    int req = MPI_THREAD_FUNNELED;
    int prov = -1;
    MPI_Init_thread(NULL, NULL, req, &prov);
    assert(prov == req);

    ::testing::InitGoogleTest(&argc, argv);
    
    if (argc > 1)
    {
        VERB = atoi(argv[1]);
    }

    const int return_flag = RUN_ALL_TESTS();

    MPI_Finalize();
    return return_flag;
}
