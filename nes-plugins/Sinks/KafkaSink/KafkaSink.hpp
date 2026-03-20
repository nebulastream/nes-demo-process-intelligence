/*
    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        https://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
*/

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <string_view>
#include <unordered_map>

#include <librdkafka/rdkafkacpp.h>

#include <Configurations/Descriptor.hpp>
#include <Runtime/TupleBuffer.hpp>
#include <Sinks/Sink.hpp>
#include <Sinks/SinkDescriptor.hpp>
#include <Util/Logger/Formatter.hpp>
#include <BackpressureChannel.hpp>
#include <PipelineExecutionContext.hpp>

namespace NES
{
class KafkaSink : public Sink
{
public:
    static constexpr std::string_view NAME = "Kafka";

    static constexpr uint64_t CONNECTION_TIMEOUT_IN_MILLISECONDS = uint64_t{1000};
    static constexpr uint64_t TERMINATION_TIMEOUT_IN_MILLISECONDS = uint64_t{10000};

    explicit KafkaSink(BackpressureController backpressureController, const SinkDescriptor& sinkDescriptor);
    ~KafkaSink() override = default;

    KafkaSink(const KafkaSink&) = delete;
    KafkaSink& operator=(const KafkaSink&) = delete;
    KafkaSink(KafkaSink&&) = delete;
    KafkaSink& operator=(KafkaSink&&) = delete;

    void start(PipelineExecutionContext& pipelineExecutionContext) override;
    void execute(const TupleBuffer& inputTupleBuffer, PipelineExecutionContext& pipelineExecutionContext) override;
    void stop(PipelineExecutionContext& pipelineExecutionContext) override;

protected:
    std::ostream& toString(std::ostream& os) const override;

private:
    class DeliveryReportCallback : public RdKafka::DeliveryReportCb
    {
    public:
        void dr_cb(RdKafka::Message& message) override;
    };

    DeliveryReportCallback deliveryReportCallback;
    std::unique_ptr<RdKafka::Producer> producer;

    std::string brokers; /// comma-separated list of host or host:port
    std::string topic; /// name of kafka topic
    uint64_t retryLimit;
};

struct ConfigParametersKafkaSink
{
    static inline const DescriptorConfig::ConfigParameter<std::string> OUTPUT_FORMAT{
        "output_format",
        "JSON",
        [](const std::unordered_map<std::string, std::string>& config) { return DescriptorConfig::tryGet(OUTPUT_FORMAT, config); }};
    static inline const DescriptorConfig::ConfigParameter<std::string> BROKERS{
        "brokers",
        std::nullopt,
        [](const std::unordered_map<std::string, std::string>& config) { return DescriptorConfig::tryGet(BROKERS, config); }};
    static inline const DescriptorConfig::ConfigParameter<std::string> TOPIC{
        "topic",
        std::nullopt,
        [](const std::unordered_map<std::string, std::string>& config) { return DescriptorConfig::tryGet(TOPIC, config); }};
    static inline const DescriptorConfig::ConfigParameter<uint64_t> RETRY_LIMIT{
        "retry_limit",
        uint64_t{5},
        [](const std::unordered_map<std::string, std::string>& config) { return DescriptorConfig::tryGet(RETRY_LIMIT, config); }};

    static inline std::unordered_map<std::string, DescriptorConfig::ConfigParameterContainer> parameterMap
        = DescriptorConfig::createConfigParameterContainerMap(OUTPUT_FORMAT, BROKERS, TOPIC, RETRY_LIMIT);
};


}

FMT_OSTREAM(NES::KafkaSink);
