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

/**
 * We send a lot of messages to next rank, starting with big ones and then small ones
 * This gives a change for small messages to sneak in before the big, and should make sure the ordering is properly respected
 */
TEST(ActiveMsgLarge, bigToSmall)
{
    const size_t break_size = (1 << 21); // Larger than 1MB but much smaller than 2^31
    
    // Various distributions of sizes
    // Accross the spectrum
    vector<size_t> sizes_0 = {32*break_size,    16*break_size,  8*break_size, 
                              4*break_size,     2*break_size,   break_size, 
                              break_size/2,     break_size/4,   break_size/8, 
                              break_size/16,    break_size/32,  break_size/64, 
                              break_size/128,   break_size/256, break_size/512};
    // Very small and very large
    // This make sure that the small ones (which are sent last) are (usually) completed first
    // So this stresses the code to verify the ordering
    vector<size_t> sizes_1 = {128*break_size, 1*break_size, break_size/16, break_size/1024};
    // Around break_size
    vector<size_t> sizes_2 = {4*break_size, 4*break_size, 4*break_size, 4*break_size, 4*break_size,
                              2*break_size, 2*break_size, 2*break_size, 2*break_size, 2*break_size,
                              1*break_size, 1*break_size, 1*break_size, 1*break_size, 1*break_size,
                              break_size/2, break_size/2, break_size/2, break_size/2, break_size/2, 
                              break_size/4, break_size/4, break_size/4, break_size/4, break_size/4};
    
    vector<vector<size_t>> sizes_all = {sizes_0, sizes_1, sizes_2};

    for(auto& sizes: sizes_all) {

        Communicator comm(VERB, break_size);

        int N = sizes.size();

        // unsigned char to make sure the wrap around is well defined (the -k before need to be well defined)
        vector<unsigned char*> send_buffers(N * N);
        vector<unsigned char*> recv_buffers(N * N);
        vector<unsigned char*> header_buffers(N * N);

        int headers_rcvd = 0;
        int bodies_rcvd = 0;
        const int expected = N * N;

        for(int i = 0; i < N; i++) {
            for(int j = 0; j < N; j++) {
                int k = i + j * N;
                size_t header_size = sizes[i];
                size_t body_size = sizes[j];

                // malloc seems _overall_ faster than std::vector somehow
                header_buffers.at(k) = (unsigned char*)malloc(header_size);
                *(header_buffers.at(k)) = (unsigned char)(- k);
                *(header_buffers.at(k)+header_size-1) = (unsigned char)(- k);

                send_buffers.at(k) = (unsigned char*)malloc(body_size);
                *(send_buffers.at(k)) = (unsigned char)k;
                *(send_buffers.at(k)+body_size-1) = (unsigned char)k;

                recv_buffers.at(k) = (unsigned char*)malloc(body_size);
                *(recv_buffers.at(k)) = (unsigned char)0;
                *(recv_buffers.at(k)+body_size-1) = (unsigned char)0;
            }
        }

        auto am = comm.make_large_active_msg(
            [&](int& i, int& j, view<unsigned char>& header) {
                int k = i + j * N;

                size_t header_size = sizes.at(i);
                EXPECT_EQ(header.size(), header_size);
                EXPECT_EQ(*(header.begin()), (unsigned char)(- k));
                EXPECT_EQ(*(header.end()-1), (unsigned char)(- k));

                // Data is there
                size_t body_size = sizes.at(j);
                EXPECT_EQ(*(recv_buffers[k]), (unsigned char)k); 
                EXPECT_EQ(*(recv_buffers[k] + body_size - 1), (unsigned char)k);

                bodies_rcvd++;
            },
            [&](int& i, int& j, view<unsigned char>& header) {
                int k = i + j * N;

                size_t header_size = sizes.at(i);
                EXPECT_EQ(header.size(), header_size);
                EXPECT_EQ(*(header.begin()), (unsigned char)(- k));
                EXPECT_EQ(*(header.end()-1), (unsigned char)(- k));

                // Data is not there
                size_t body_size = sizes.at(j);
                EXPECT_EQ(*(recv_buffers[k]), (unsigned char)0); 
                EXPECT_EQ(*(recv_buffers[k] + body_size - 1), (unsigned char)0);

                unsigned char* ptr = recv_buffers.at(k);
                headers_rcvd++;
                return ptr;
            });

        int dest = (comm_rank() + 1) % (comm_size());
        for(int i = 0; i < N; i++) {
            for(int j = 0; j < N; j++) {
                int k = i + j * N;
                size_t header_size = sizes.at(i);
                size_t body_size = sizes.at(j);
                auto body   = view<unsigned char>(send_buffers.at(k), body_size);
                auto header = view<unsigned char>(header_buffers.at(k), header_size);
                am->send_large(dest, body, i, j, header);
            }
        }

        while ( (!comm.is_done()) || (bodies_rcvd != expected) ) {
            comm.progress();
        }

        EXPECT_EQ(bodies_rcvd, expected);
        EXPECT_EQ(headers_rcvd, expected);
        for(int i = 0; i < N; i++) {
            for(int j = 0; j < N; j++) {
                int k = i + j * N;
                size_t body_size = sizes.at(j);
                EXPECT_EQ(*(recv_buffers.at(k)), (unsigned char)k);
                EXPECT_EQ(*(recv_buffers.at(k) + body_size - 1), (unsigned char)k);
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);

        for(auto v: send_buffers) free(v);
        for(auto v: recv_buffers) free(v);
        for(auto v: header_buffers) free(v);
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
