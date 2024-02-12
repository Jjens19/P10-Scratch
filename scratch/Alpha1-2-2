#include <iostream>
#include <fstream>
#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/applications-module.h"
#include "ns3/bridge-module.h"
#include "ns3/csma-module.h"
#include "ns3/internet-module.h"
#include <vector>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("Project");

uint32_t g_packetCount = 0;
uint32_t g_bytesSent = 0;
uint32_t g_ackReceived = 0;
Time totalRtt = Seconds(0.0);  // Total RTT for all received packets
Time lastPacketTime;  // Keep track of the timestamp of the most recent packet

// Declare the RTT estimator globally
Ptr<RttMeanDeviation> rttEstimator;

void resetVals()
{
    g_packetCount = 0;
    g_bytesSent = 0;
    g_ackReceived = 0;
    totalRtt = Seconds(0.0);
}

void TxPacketsTrace(Ptr<const Packet> packet)
{
    g_packetCount++;
    g_bytesSent += packet->GetSize(); // Accumulate the size of each sent packet
    lastPacketTime = Simulator::Now();
    rttEstimator->Measurement(lastPacketTime);
}

void RxPacketsTrace(Ptr<const Packet> packet, const Address &addr)
{
    g_ackReceived++;
    totalRtt += (Simulator::Now() - lastPacketTime);  // Update total RTT
    rttEstimator->Measurement(Simulator::Now());
}

int main(int argc, char *argv[])
{
    RngSeedManager::SetSeed(1);
    RngSeedManager::SetRun(1);

    #if 1
    LogComponentEnable("Project", LOG_LEVEL_INFO);
    #endif

    CommandLine cmd;
    cmd.Parse(argc, argv);

    NS_LOG_INFO("Create nodes.");
    NodeContainer terminals;
    terminals.Create(10);

    Config::SetDefault("ns3::TcpL4Protocol::SocketType", StringValue("ns3::TcpLinuxReno"));

    NodeContainer csmaSwitch;
    csmaSwitch.Create(2);

    NS_LOG_INFO("Build Topology");
    CsmaHelper csma;
    csma.SetChannelAttribute("DataRate", DataRateValue(DataRate("1000kbps")));
    csma.SetChannelAttribute("Delay", TimeValue(MilliSeconds(2)));

    NetDeviceContainer terminalDevices;
    NetDeviceContainer switch1Devices;
    NetDeviceContainer switch2Devices;

    for (int i = 0; i < 5; i++)
    {
        NetDeviceContainer link = csma.Install(NodeContainer(terminals.Get(i), csmaSwitch.Get(0)));
        terminalDevices.Add(link.Get(0));
        switch1Devices.Add(link.Get(1));
    }

    for (int i = 5; i < 10; i++)
    {
        NetDeviceContainer link = csma.Install(NodeContainer(terminals.Get(i), csmaSwitch.Get(1)));
        terminalDevices.Add(link.Get(0));
        switch2Devices.Add(link.Get(1));
    }

    NetDeviceContainer link = csma.Install(NodeContainer(csmaSwitch.Get(0), csmaSwitch.Get(1)));
    switch1Devices.Add(link.Get(0));
    switch2Devices.Add(link.Get(1));

    Ptr<Node> switchNode = csmaSwitch.Get(0);
    BridgeHelper bridge;
    bridge.Install(switchNode, switch1Devices);

    switchNode = csmaSwitch.Get(1);
    bridge.Install(switchNode, switch2Devices);

    InternetStackHelper internet;
    internet.Install(terminals);

    NS_LOG_INFO("Assign IP Addresses.");
    Ipv4AddressHelper ipv4;
    ipv4.SetBase("10.1.1.0", "255.255.255.0");
    ipv4.Assign(terminalDevices);

    uint16_t port = 9;

    OnOffHelper onoff0("ns3::TcpSocketFactory", Address(InetSocketAddress(Ipv4Address("10.1.1.1"), port)));
    onoff0.SetConstantRate(DataRate("80kb/s"));
    
    OnOffHelper onoff1 ("ns3::TcpSocketFactory", Address(InetSocketAddress(Ipv4Address("10.1.1.2"), port))); //Create destination for N1
    onoff1.SetConstantRate(DataRate("80kb/s"));

    OnOffHelper onoff2 ("ns3::TcpSocketFactory", Address(InetSocketAddress(Ipv4Address("10.1.1.3"), port))); //Create destination for N2
    onoff2.SetConstantRate(DataRate("80kb/s"));

    OnOffHelper onoff3 ("ns3::TcpSocketFactory", Address(InetSocketAddress(Ipv4Address("10.1.1.4"), port))); //Create destination for N3
    onoff3.SetConstantRate(DataRate("80kb/s"));

    OnOffHelper onoff4 ("ns3::TcpSocketFactory", Address(InetSocketAddress(Ipv4Address("10.1.1.5"), port))); //Create destination for N4
    onoff4.SetConstantRate(DataRate("80kb/s"));

    OnOffHelper onoff5 ("ns3::TcpSocketFactory", Address(InetSocketAddress(Ipv4Address("10.1.1.6"), port))); //Create destination for N5
    onoff5.SetConstantRate(DataRate("80kb/s"));

    OnOffHelper onoff6 ("ns3::TcpSocketFactory", Address(InetSocketAddress(Ipv4Address("10.1.1.7"), port))); //Create destination for N6
    onoff6.SetConstantRate(DataRate("80kb/s"));

    OnOffHelper onoff7 ("ns3::TcpSocketFactory", Address(InetSocketAddress(Ipv4Address("10.1.1.8"), port))); //Create destination for N7
    onoff7.SetConstantRate(DataRate("80kb/s"));

    OnOffHelper onoff8 ("ns3::TcpSocketFactory", Address(InetSocketAddress(Ipv4Address("10.1.1.9"), port))); //Create destination for N8
    onoff8.SetConstantRate(DataRate("80kb/s"));

    OnOffHelper onoff9 ("ns3::TcpSocketFactory", Address(InetSocketAddress(Ipv4Address("10.1.1.10"), port))); //Create destination for N9
    onoff9.SetConstantRate(DataRate("80kb/s"));

    // Create packetsink helper
    PacketSinkHelper sink("ns3::TcpSocketFactory", Address(InetSocketAddress(Ipv4Address::GetAny(), port)));

    // Application sends from N9 to N0
    ApplicationContainer app1 = onoff0.Install(terminals.Get(9));
    app1.Get(0)->TraceConnectWithoutContext("Tx", MakeCallback(&TxPacketsTrace));
    app1.Start(Seconds(1.0));
    app1.Stop(Seconds(65.0));

    ApplicationContainer sinkApp = sink.Install(terminals.Get(0));
    sinkApp.Get(0)->TraceConnectWithoutContext("Rx", MakeCallback(&RxPacketsTrace));

    // Application sends from N7 to N3
    ApplicationContainer app2 = onoff3.Install(terminals.Get(7));
    app2.Start(Seconds(1.0));
    app2.Stop(Seconds(15.0)); 

    app2 = sink.Install(terminals.Get(3));

    NS_LOG_INFO("Configure Tracing.");

    AsciiTraceHelper ascii;
    csma.EnableAsciiAll("Project");
    csma.EnablePcapAll("Project", false);

    // Create an instance of RttMeanDeviation
    rttEstimator = CreateObject<RttMeanDeviation>();

    NS_LOG_INFO("Run Simulation.");
    do
    {
        resetVals();
        Simulator::Stop(Seconds(5.0));
        Simulator::Run();

        // Calculate average RTT
        double averageRtt = 0.0;
        if (g_ackReceived > 0)
        {
            averageRtt = totalRtt.GetMilliSeconds() / g_ackReceived;
        }

        // Print average RTT
        //NS_LOG_INFO("Average RTT for received packets: " << averageRtt << " seconds");
        NS_LOG_UNCOND(g_packetCount << "," << g_ackReceived << "," << g_bytesSent << "," << averageRtt);

    } while (g_packetCount > 0 || g_ackReceived > 0 || g_bytesSent > 0);

    Simulator::Destroy();

    NS_LOG_INFO("Done.");

    return 0;
}

