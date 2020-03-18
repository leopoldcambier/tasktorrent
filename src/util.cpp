#include "util.hpp"

namespace ttor {

    double elapsed(timer start, timer end) {
        std::chrono::duration<double> elapsed_seconds = end - start;
        return elapsed_seconds.count();
    }

    double time(timer timer) {
        return std::chrono::duration<double>(timer.time_since_epoch()).count();
    }

    timer wctime() {
        return std::chrono::high_resolution_clock::now();
    }

    std::string get_name_me() {
        char name[50];
        int err = gethostname(name, 50);
        if(err == 0) {
            std::string name_(name);
            return name_;
        } else {
            return "ErrorNoName";
        }
    }

    // Logging

    Event::Event(std::string name_) {
        start = wctime();
        name = name_;
    }
    Event::Event() {
        start = wctime();
        name = "_";
    }
    void Event::record() {
        end = wctime();
    }

    Logger::Logger(int N) : origin(wctime()), printed_warning(false), i(0), events(N) {}
    Logger::Logger() : origin(wctime()), printed_warning(false), i(0), events(0) {}
    void Logger::add_event(std::unique_ptr<Event> e) {
        int id = (i++); // atomic
        if(id < int(events.size())) {
            events[id] = move(e);
        } else {
            if(! printed_warning.load()) {
                printf("Logger::add_event buffer full\n");
                printed_warning.store(true);
            }
        }
    }
    void Logger::record(std::unique_ptr<Event> e) {
        e->record();
        add_event(std::move(e));
    }
    int Logger::n_events() const {
        return std::min(int(events.size()), i.load());
    }
    bool Logger::was_full() const {
        return printed_warning.load();
    }
    const std::vector<std::unique_ptr<Event>>& Logger::get_events() const {
        return events;
    }
    timer Logger::get_origin() const {
        return origin;
    }

    // Deps Tracking
    DepsEvent::DepsEvent(std::string s, std::string o) {
        self_name = s;
        out_deps_name = o;
    }
    DepsEvent::DepsEvent() {}
    DepsLogger::DepsLogger(int N) : i(0), printed_warning(false), events(N) {}
    DepsLogger::DepsLogger() : i(0), printed_warning(false), events(0) {}
    void DepsLogger::add_event(std::unique_ptr<DepsEvent> e) {
        int id = (i++); // atomic
        if(id < int(events.size())) {
            events[id] = move(e);
        } else {
            printed_warning.store(true);
            printf("DepsLogger::add_event buffer full\n");
        }
    }

    bool DepsLogger::was_full() const {
        return printed_warning.load();
    }

    std::ostream &operator << (std::ostream &os, ttor::DepsLogger &t)  {
        if(t.was_full()) {
            printf("Warning: buffer was full at some point during the profiling\n");
        }
        // os << "digraph g{\n";
        for(int i = 0; i < std::min(int(t.events.size()), t.i.load()); i++) {
            auto e = t.events[i].get();
            // os << e->self_name << " -> " << e->out_deps_name << ";\n";
            os << e->self_name << "," << e->out_deps_name << "\n";
        }
        // os << "}\n";
        return os;
    }

    std::ostream &operator << (std::ostream &os, ttor::Logger &t) {
        if(t.was_full()) {
            printf("Warning: buffer was full at some point during the profiling\n");
        }
        os << std::setprecision(64);
        for(int i = 0; i < t.n_events(); i++) {
            const auto& e = t.get_events()[i];
            os << e->name << "," << ttor::time(e->start) << "," << ttor::time(e->end) << "\n";
        }
        return os;
    }
    
}