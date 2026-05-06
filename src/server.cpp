#include "../include/server.h"

#include "../include/dto/sort_request_dto.h"
#include "../include/sorter.h"

#include <exception>
#include <stdexcept>
#include <string>
#include <vector>

crow::response json_response(int statusCode, crow::json::wvalue json) {
    crow::response response(statusCode, json);
    response.set_header("Content-Type", "application/json");
    response.set_header("Access-Control-Allow-Origin", "*");
    response.set_header("Access-Control-Allow-Headers", "Content-Type");
    response.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
    return response;
}

crow::response error_response(int statusCode, const std::string& message) {
    crow::json::wvalue json;
    json["success"] = false;
    json["message"] = message;
    return json_response(statusCode, json);
}

sort_request_dto parse_sort_request(const crow::request& request) {
    auto body = crow::json::load(request.body);

    if (!body || !body.has("n") || !body.has("u") || !body.has("distribution")) {
        throw std::runtime_error("Body must include n, u and distribution.");
    }

    sort_request_dto dto;
    dto.n = body["n"].i();
    dto.u = body["u"].i();
    dto.distribution = static_cast<std::string>(body["distribution"]);

    return dto;
}

void server::run(int port) {
    setUpRoutes();
    app.loglevel(crow::LogLevel::Warning);
    app.port(port).multithreaded().run();
}

void server::setUpRoutes() {
    crow::mustache::set_global_base("../templates");

    CROW_ROUTE(app, "/")([]() {
        auto page = crow::mustache::load("index.html");
        return page.render();
    });

    CROW_ROUTE(app, "/test")([]() {
        auto page = crow::mustache::load("test.html");
        return page.render();
    });

    CROW_ROUTE(app, "/health")([] {
        return "ok";
    });

    CROW_ROUTE(app, "/sort").methods(crow::HTTPMethod::Options)([] {
        return json_response(204, crow::json::wvalue());
    });

    CROW_ROUTE(app, "/compare").methods(crow::HTTPMethod::Options)([] {
        return json_response(204, crow::json::wvalue());
    });

    CROW_ROUTE(app, "/sort").methods(crow::HTTPMethod::Post)(
        [](const crow::request& request) -> crow::response {
            try {
                const char* algorithmParam = request.url_params.get("algorithm");
                std::string algorithm = algorithmParam ? algorithmParam : "normal";
                sort_request_dto sortRequest = parse_sort_request(request);
                sorter sorter;

                if (algorithm == sorter.neuralSortName || algorithm == sorter.normalSortName) {
                    sort_result_dto result = sorter.sort(sortRequest, algorithm);
                    return json_response(200, result.toJson());
                }

                return error_response(400, "algorithm must be normal or neural.");
            } catch (const std::exception& error) {
                return error_response(400, error.what());
            }
        }
    );

    CROW_ROUTE(app, "/compare").methods(crow::HTTPMethod::Post)(
        [](const crow::request& request) -> crow::response {
            try {
                sort_request_dto sortRequest = parse_sort_request(request);
                sorter sorter;
                std::vector<sort_result_dto> results = sorter.compare(sortRequest);

                crow::json::wvalue response;
                for (size_t i = 0; i < results.size(); ++i) {
                    response[static_cast<unsigned>(i)] = results[i].toJson();
                }

                return json_response(200, response);
            } catch (const std::exception& error) {
                return error_response(400, error.what());
            }
        }
    );
}
