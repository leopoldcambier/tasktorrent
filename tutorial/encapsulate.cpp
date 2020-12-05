#include "tasktorrent/tasktorrent.hpp"

#include <iostream>
#include <map>

using namespace std;
using namespace ttor;

using int2 = std::array<int,2>;

/**
 * This algorithm goal is to compute something (yes)
 * based on some index `k`
 * 
 * Some rank (called ORIGIN in the following) has some big computation to do
 * ORIGIN will ask other ranks, called HELPERS, for help doing that and 
 * eventually those ranks will send back their answer to ORIGIN
 * 
 * ORIGIN will then fulfill back some dependency on the parent task
 * and is then done
 */
struct VeryComplexAlgorithm {

    /** Usual stuff */
    const int rank;
    const int nranks;
    const int n_threads_;
    Communicator* comm_;
    Threadpool* tp_;

    /** 
     * When the VeryComplexAlgorithm task is done, 
     * ORIGIN will fulfill dependency `k` on this
     * task flow
     */
    Taskflow<int>* tf_notify_done_;

    /**
     * This is the actual very complex algorithm
     * This can really be anything
     * Here int2 = (ORIGIN, k)
     */
    Taskflow<int2> tf_compute_;

    /**
     * This is used to track completion and 
     * fulfill the dependency on tf_notify_done_
     * ORIGIN expects nranks answers back from the
     * other ranks
     * So this track this count of dependencies
     * Whenever it's done, it will fulfill the dependency on
     * tf_notify_done_
     */
    Taskflow<int> tf_notify_count_;

    /**
     * This first active message is send by ORIGIN
     * to the other ranks to ask for help
     */
    ActiveMsg<int,int,int>* ask_help_;

    /**
     * Whenever the other ranks are done, they
     * send back their block to ORIGIN
     */
    ActiveMsg<int,int,int>* send_back_matrix_;

    /**
     * This is some scratch workspace to store the intermediate calculations
     * This would normally be more complicated obviously
     * 
     * Every rank receives requests for help from everyone, so we store them all
     * contiguously
     * 
     * In realistic use case, this should probably be more dynamic 
     * (need a lock/mutex then), like a map with one entry for every `k`
     */
    std::vector<int> workspace_; // workspace_[ORIGIN] = sub_matrix

    /**
     * This is the input and output data
     * Every rank has one computation (one `k`) to do, so we just have one
     * 
     * In realistic use case, this should be more dynamic, 
     * like a map with as many entries as there are `k` 
     */ 
    std::vector<int>* input_output_;

    VeryComplexAlgorithm(Communicator* comm, Threadpool* tp, Taskflow<int>* tf_notify_done) : rank(comm->comm_rank()), 
        nranks(comm->comm_size()), n_threads_(tp->size()), comm_(comm), tp_(tp), tf_notify_done_(tf_notify_done), tf_compute_(tp), tf_notify_count_(tp) {
            
            workspace_.resize(nranks);

        // From origin to helper ranks
        ask_help_ = comm->make_active_msg([&](int& origin, int& k, int& sub_matrix){
            workspace_[origin] = sub_matrix;
            tf_compute_.fulfill_promise({origin, k});
        });

        // From helper ranks to origin
        send_back_matrix_ = comm->make_active_msg([&](int& helper, int& k, int& sub_matrix){
            // This is writing into the parent task memory location directly
            (*input_output_)[helper] = sub_matrix;
            tf_notify_count_.fulfill_promise(k);
        });

        // This is the actual algorithm
        // where HELPER is helping ORIGIN into computing some stuff, 
        // called "sub_matrix" here
        tf_compute_.set_task([&](int2 origin_k) {
            int origin = origin_k[0];
            int helper = rank;
            int k      = origin_k[1];
            int magic_number = 0123; // Quizz: what is this equal to ? Not 123 :-)
            // Do some complex computation
            // This will be where our complex algorithm would normally take place
            printf("[%d] VeryComplexAlgorithm: Helper computing k=%d\n", rank, k);
            int sub_matrix = workspace_[origin];
            sub_matrix = sub_matrix * 314 / 271 + magic_number;
            // Send back the answer
            send_back_matrix_->send(origin, helper, k, sub_matrix);
        })
        .set_indegree([&](int2 origin_k) {
            return 1;
        })
        .set_mapping([&](int2 origin_k) {
            return (origin_k[0] % n_threads_);
        });

        // We need to wait for all the pieces to come back from the HELPER
        // to ORIGIN
        // Whenever this is the case, we can fulfill the parent task
        // to indicate we are done
        tf_notify_count_.set_task([&](int k) {
            tf_notify_done_->fulfill_promise(k);
        })
        .set_indegree([&](int k) {
            return nranks;
        })
        .set_mapping([&](int k) {
            return (k % n_threads_);
        });
    }

    // Entry point
    void run(int k, std::vector<int>* matrix) {
        printf("[%d] VeryComplexAlgorithm: Starting complex algorithm for k = %d\n", rank, k);
        assert(matrix->size() == nranks);
        input_output_ = matrix;
        int origin = rank;
        for(int helper = 0; helper < nranks; helper++) {
            int sub_matrix = (*input_output_)[helper];
            ask_help_->send(helper, origin, k, sub_matrix);
        }
    }
};

void algorithm()
{
    const int rank = comm_rank();
    const int n_ranks = comm_size();
    const int n_threads = 2;

    printf("[%d] Hello from %s\n", rank, processor_name().c_str());

    Communicator comm(MPI_COMM_WORLD);
    Threadpool tp(n_threads, &comm);
    Taskflow<int> tf_produce(&tp);
    Taskflow<int> tf_consume(&tp);

    VeryComplexAlgorithm algo(&comm, &tp, &tf_consume);

    std::vector<int> matrix(n_ranks, -1);

    tf_produce.set_task([&](int k) {
        printf("[%d] Starting big computation on block %d\n", rank, k);
        algo.run(k, &matrix);
    })
    .set_indegree([&](int k) {
        return 1;
    })
    .set_mapping([&](int k) {
        return (k % n_threads);
    });

    tf_consume.set_task([&](int k) {
        printf("[%d] Block %d is computed and ready now!\n", rank, k);
    })
    .set_indegree([&](int k) {
        return 1;
    })
    .set_mapping([&](int k) {
        return (k % n_threads);
    });

    tf_produce.fulfill_promise(rank);

    tp.join();
    
    printf("[%d] matrix[0] = %d (> -1)\n", rank, matrix[0]);

}

int main(int argc, char **argv)
{
    int req = MPI_THREAD_FUNNELED;
    int prov = -1;
    MPI_Init_thread(NULL, NULL, req, &prov);
    assert(prov == req);

    algorithm();

    MPI_Finalize();
}
