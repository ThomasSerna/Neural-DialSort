#ifndef NEURAL_DIALSORT_SERVER_H
#define NEURAL_DIALSORT_SERVER_H

#include "crow_all.h"
#include "sorter.h"

class server {
public:
    void run(int port = 8080);

private:
    crow::SimpleApp app;
    sorter sortService;

    void setUpRoutes();
};

#endif // NEURAL_DIALSORT_SERVER_H
