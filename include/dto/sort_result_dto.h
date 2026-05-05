#pragma once

#include "crow_all.h"

#include <cstdint>
#include <string>
#include <vector>

struct sort_result_dto {
    std::vector<int64_t> sortedVector;
    std::string algorithmUsed;
    int64_t n = 0;
    int64_t u = 0;
    bool success = false;
    double elapsedMs = 0.0;

    crow::json::wvalue toJson() const {
        crow::json::wvalue json;

        // json["sortedVector"] = sortedVector;
        json["algorithmUsed"] = algorithmUsed;
        json["n"] = n;
        json["u"] = u;
        json["success"] = success;
        json["elapsedMs"] = elapsedMs;

        return json;
    }
};
