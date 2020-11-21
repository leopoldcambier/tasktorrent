#include <vector>
#include <gtest/gtest.h>

#include <mpi.h>

#include "tasktorrent/tasktorrent.hpp"

using namespace std;
using namespace ttor;

int VERB = 0;

/** This works with MPI but not UPCXX because of UPCXX's design **/
TEST(Communicator, ActiveMessages_Ref)
{
    int rank = comms_world_rank();
    int nranks = comms_world_size();
    if(nranks < 2) {
        printf("Need at least 2 ranks\n");
        return;
    }
    auto comm = make_communicator_world(VERB);
    int rcvd = 0;
    int expt = 2;
    auto am1 = comm->make_active_msg([&](int&){ rcvd++; });

    if (rank == 0)
    {
        for (int dest = 1; dest < comms_world_size(); dest++)
        {
            int one = 1;
            int& one_ref = one;

            am1->blocking_send(dest, one);
            am1->blocking_send(dest, one_ref);
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

TEST(ActiveMsgBody, multipleBodiesBreakSize)
{
    vector<double> sizes = {0.001, 0.1, 0.9, 1.1, 1.5, 4.0, 8.0}; 
    const size_t break_size = (1 << 22); // Larger than 1MB but smaller than 2^31
    for(auto s_header: sizes) {
        for(auto s_body: sizes) {
            if(VERB) printf("Size factor = %e, %e=======================\n", s_header, s_body);
            Communicator comm(MPI_COMM_WORLD, VERB, break_size);
            int done = 0;
            int rcvd = 0;
            int expected = 1;
            int completed = 0;
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
                },
                [&](view<char>&){
                    completed++;
                });

            auto body = view<char>(send_buffer, body_size);
            auto header = view<char>(header_buffer, header_size);
            int dest = (comms_world_rank() + 1) % (comms_world_size());
            am->send_large(dest, body, header);

            while ( (!comm.is_done()) || (done != expected) ) {
                comm.progress();
            }
            
            EXPECT_EQ(done, expected);
            EXPECT_EQ(rcvd, expected);
            EXPECT_EQ(completed, 1);
            comms_world_barrier();
            free(send_buffer);
            free(recv_buffer);
            free(header_buffer);
        }
    }
}

void test_sizes(vector<size_t> sizes, size_t break_size) {
    Communicator comm(MPI_COMM_WORLD, VERB, break_size);

    int N = sizes.size();

    // unsigned char to make sure the wrap around is well defined (the -k before need to be well defined)
    vector<unsigned char*> send_buffers(N * N);
    vector<unsigned char*> recv_buffers(N * N);
    vector<unsigned char*> header_buffers(N * N);

    int headers_rcvd = 0;
    int bodies_rcvd = 0;
    const int expected = N * N;
    int completed = 0;

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
        },
        [&](int&, int&, view<unsigned char>&) {
            completed++;
        });

    int dest = (comms_world_rank() + 1) % (comms_world_size());
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
    EXPECT_EQ(completed, N*N);
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            int k = i + j * N;
            size_t body_size = sizes.at(j);
            EXPECT_EQ(*(recv_buffers.at(k)), (unsigned char)k);
            EXPECT_EQ(*(recv_buffers.at(k) + body_size - 1), (unsigned char)k);
        }
    }

    comms_world_barrier();

    for(auto v: send_buffers) free(v);
    for(auto v: recv_buffers) free(v);
    for(auto v: header_buffers) free(v);
}

/**
 * We send a lot of messages to next rank, starting with big ones and then small ones
 * This gives a change for small messages to sneak in before the big, and should make sure the ordering is properly respected
 */
TEST(BigToSmall, large)
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

    test_sizes(sizes_0, break_size);
    test_sizes(sizes_1, break_size);
    test_sizes(sizes_2, break_size);
}

/**
 * We send a lot of messages to next rank, starting with big ones and then small ones
 * This gives a change for small messages to sneak in before the big, and should make sure the ordering is properly respected
 */
TEST(BigToSmall, small)
{
    const size_t break_size = (1 << 21); // Larger than 1MB but much smaller than 2^31
    
    // Various distributions of sizes
    // Accross the spectrum
    vector<size_t> sizes_0 = {4*break_size,     2*break_size,   break_size, 
                              break_size/2,     break_size/4,   break_size/8, 
                              break_size/16,    break_size/32,  break_size/64, 
                              break_size/128,   break_size/256, break_size/512};
    // Very small and very large
    // This make sure that the small ones (which are sent last) are (usually) completed first
    // So this stresses the code to verify the ordering
    vector<size_t> sizes_1 = {4*break_size, 1*break_size, break_size/16, break_size/1024};
    // Around break_size
    vector<size_t> sizes_2 = {4*break_size, 4*break_size,
                              2*break_size, 2*break_size,
                              1*break_size, 1*break_size,
                              break_size/2, break_size/2, 
                              break_size/4, break_size/4};

    test_sizes(sizes_0, break_size);
    test_sizes(sizes_1, break_size);
    test_sizes(sizes_2, break_size);
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
        [&](const view<char> &p) {
            const char* buffer = p.data();
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
    am->send( (comms_world_rank() + 1) % (comms_world_size()) , v);

    while ( (!comm.is_done()) || (done != expected) ) {
        comm.progress();
    }
    
    EXPECT_EQ(done, expected);
    comms_world_barrier();
    free(buffer);
}

INSTANTIATE_TEST_SUITE_P(
    Ttor, BreakSize,
    ::testing::Combine(
        ::testing::Values(0.001, 0.01, 0.1, 0.5, 0.9, 1.1, 1.2, 1.3, 1.5, 2.0, 4.0, 5.5, 6.0, 7.8)
    )
);

// This checks that using a custom communicator
// allows to interleave MPI comms from different communicators
TEST(Communicator, newMPICommunicator) {
    MPI_Comm new_world;
    MPI_Comm_dup(MPI_COMM_WORLD, &new_world);

    const int n_ranks = mpi_comm_size(MPI_COMM_WORLD);
    const int rank = mpi_comm_rank(MPI_COMM_WORLD);
    const int other = (rank + 1) % n_ranks;
    int expected = 1;
    int done = 0;

    EXPECT_EQ(n_ranks, mpi_comm_size(new_world));
    EXPECT_EQ(rank,    mpi_comm_rank(new_world));

    Communicator comm(new_world, VERB);
    auto am = comm.make_active_msg([&]() { done++; });

    EXPECT_EQ(comm.comm_size(), n_ranks);
    EXPECT_EQ(comm.comm_rank(), rank);

    MPI_Request req_0, req_1;
    int send = 3;
    int recv = 0;
    MPI_Isend(&send, 1, MPI_INT, other, 0, MPI_COMM_WORLD, &req_0);
    am->send(other);

    {
        int send_2 = 1;
        int recv_2 = 0;
        MPI_Allreduce(&send_2, &recv_2, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        EXPECT_EQ(recv_2, n_ranks);
    }

    while ( (!comm.is_done()) || (done != expected) ) {
        comm.progress();
    }

    MPI_Irecv(&recv, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &req_1);
    MPI_Wait(&req_0, MPI_STATUS_IGNORE);
    MPI_Wait(&req_1, MPI_STATUS_IGNORE);
    EXPECT_EQ(recv, 3);

    comms_world_barrier();
    MPI_Comm_free(&new_world);
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
