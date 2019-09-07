#ifndef __TTOR_UTIL_HPP__
#define __TTOR_UTIL_HPP__

#include <sys/time.h>
#include <string>
#include <iostream>
#include <thread>
#include <atomic>
#include <vector>
#include <stdio.h>
#include <chrono>
#include <ctime>
#include <ratio>
#include <assert.h>
#include <unistd.h>
#include <iomanip>

namespace ttor
{

typedef std::chrono::time_point<std::chrono::high_resolution_clock> timer;

struct Logger;
struct Event;

double elapsed(timer start, timer end);
double time(timer timer);
timer wctime();
std::string get_name_me();

// Logging
struct Event
{
    std::string name;
    timer start;
    timer end;
    Event(std::string name_);
    Event();
    void record();
};

struct Logger
{
private:
    timer origin;
    std::atomic<bool> printed_warning;
    std::atomic<int> i;
    std::vector<std::unique_ptr<Event>> events;

public:
    Logger(int N);
    Logger();
    void add_event(std::unique_ptr<Event> e);
    void record(std::unique_ptr<Event> e);
    int n_events() const;
    bool was_full() const;
    const std::vector<std::unique_ptr<Event>> &get_events() const;
    timer get_origin() const;
};

// Deps Tracking
struct DepsEvent
{
    std::string self_name;
    std::string out_deps_name;
    DepsEvent(std::string s, std::string o);
    DepsEvent();
};
struct DepsLogger
{
    std::atomic<int> i;
    std::vector<DepsEvent> events;
    DepsLogger(int N);
    DepsLogger();
    void add_event(DepsEvent e);
};
} // namespace ttor

std::ostream &operator<<(std::ostream &os, ttor::DepsLogger &t);
std::ostream &operator<<(std::ostream &os, ttor::Logger &t);

#endif