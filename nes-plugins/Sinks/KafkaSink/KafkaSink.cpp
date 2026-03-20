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

#include <cstdint>
#include <memory>
#include <ostream>
#include <string>
#include <string_view>
#include <utility>
#include <KafkaSink.hpp>

#include <fmt/format.h>
#include <librdkafka/rdkafkacpp.h>

#include <Configurations/Descriptor.hpp>
#include <Runtime/TupleBuffer.hpp>
#include <Sinks/Sink.hpp>
#include <Sinks/SinkDescriptor.hpp>
#include <SinksParsing/BufferIterator.hpp>
#include <Util/Logger/Logger.hpp>
#include <Util/Strings.hpp>
#include <BackpressureChannel.hpp>
#include <ErrorHandling.hpp>
#include <PipelineExecutionContext.hpp>
#include <SinkRegistry.hpp>
#include <SinkValidationRegistry.hpp>

namespace NES
{
void KafkaSink::DeliveryReportCallback::dr_cb(RdKafka::Message& message)
{
    if (message.err() != 0)
    {
        NES_WARNING("Message delivery failed: {}", message.errstr());
    }
}

KafkaSink::KafkaSink(BackpressureController backpressureController, const SinkDescriptor& sinkDescriptor)
    : Sink(std::move(backpressureController))
    , brokers(sinkDescriptor.getFromConfig(ConfigParametersKafkaSink::BROKERS))
    , topic(sinkDescriptor.getFromConfig(ConfigParametersKafkaSink::TOPIC))
    , retryLimit(sinkDescriptor.getFromConfig(ConfigParametersKafkaSink::RETRY_LIMIT))
{
}

void KafkaSink::start(PipelineExecutionContext&)
{
    PRECONDITION(!producer, "Sink cannot be started twice");
    auto conf = std::unique_ptr<RdKafka::Conf>(RdKafka::Conf::create(RdKafka::Conf::CONF_GLOBAL));

    std::string error;
    if (conf->set("bootstrap.servers", brokers, error) != RdKafka::Conf::CONF_OK)
    {
        throw CannotOpenSink(error);
    }

    if (conf->set("dr_cb", &deliveryReportCallback, error) != RdKafka::Conf::CONF_OK)
    {
        throw CannotOpenSink(error);
    }

    producer.reset(RdKafka::Producer::create(conf.get(), error));
    if (!producer)
    {
        throw CannotOpenSink(error);
    }
}

void KafkaSink::execute(const TupleBuffer& inputTupleBuffer, PipelineExecutionContext& /*pipelineExecutionContext*/)
{
    PRECONDITION(inputTupleBuffer, "Invalid input buffer in KafkaSink.");

    BufferIterator iterator{inputTupleBuffer};
    std::string out;
    auto element = iterator.getNextElement();
    while (element.has_value())
    {
        const std::string_view inputView{element.value().buffer.getAvailableMemoryArea<char>().data(), element.value().contentLength};
        out += inputView;
        element = iterator.getNextElement();
    }

    auto messages = splitWithStringDelimiter<std::string>(out, "\n");
    for (auto msg : messages)
    {
        if (msg.empty())
        {
            continue;
        }

        bool retry = false;
        uint64_t retryCounter = 0;

        do
        {
            retry = false;
            const RdKafka::ErrorCode err = producer->produce(
                topic,
                RdKafka::Topic::PARTITION_UA,
                RdKafka::Producer::RK_MSG_COPY,
                msg.data(),
                msg.size(),
                nullptr,
                0,
                0,
                nullptr,
                nullptr);

            if (err != RdKafka::ERR_NO_ERROR)
            {
                if (err == RdKafka::ERR__QUEUE_FULL && retryCounter < retryLimit)
                {
                    producer->poll(CONNECTION_TIMEOUT_IN_MILLISECONDS);
                    retry = true;
                    ++retryCounter;
                }
                else
                {
                    throw SinkWriteError("Failed to send data to Kafka: {}", RdKafka::err2str(err));
                }
            }
        } while (retry);
    }
    producer->poll(0);
}

void KafkaSink::stop(PipelineExecutionContext&)
{
    if (!producer)
    {
        return;
    }

    producer->flush(TERMINATION_TIMEOUT_IN_MILLISECONDS);
    if (producer->outq_len() > 0)
    {
        NES_WARNING("{} message(s) were not delivered", producer->outq_len());
    }
    producer.reset();
}

std::ostream& KafkaSink::toString(std::ostream& os) const
{
    os << fmt::format("KafkaSink(brokers: {}, topic: {})", brokers, topic);
    return os;
}

SinkValidationRegistryReturnType RegisterKafkaSinkValidation(SinkValidationRegistryArguments sinkConfig)
{
    return DescriptorConfig::validateAndFormat<ConfigParametersKafkaSink>(std::move(sinkConfig.config), KafkaSink::NAME);
}

SinkRegistryReturnType RegisterKafkaSink(SinkRegistryArguments sinkRegistryArguments)
{
    return std::make_unique<KafkaSink>(std::move(sinkRegistryArguments.backpressureController), sinkRegistryArguments.sinkDescriptor);
}

}
