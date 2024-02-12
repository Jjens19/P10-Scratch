#include <iostream>
#include <fstream>
#include <chrono>  // Include for time-related functionality
#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/applications-module.h"
#include "ns3/bridge-module.h"
#include "ns3/csma-module.h"
#include "ns3/internet-module.h"
#include <vector>
#include "ns3/MyModules-module.h"



using namespace ns3;

NS_LOG_COMPONENT_DEFINE("Project");



int main(int argc, char *argv[])
{
    RngSeedManager::SetSeed(1);
    RngSeedManager::SetRun(1);
    
    
    float simLength = 90.0;

    #if 1
    LogComponentEnable("Project", LOG_LEVEL_INFO);
    #endif

    CommandLine cmd;
    cmd.Parse(argc, argv);

    // Record start time using C++ chrono
    auto startTime = std::chrono::high_resolution_clock::now();

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
    // Assign IP addresses.
	Ipv4AddressHelper ipv4;
	ipv4.SetBase("10.1.1.0", "255.255.255.0");
	Ipv4InterfaceContainer interfaces = ipv4.Assign(terminalDevices);

	uint16_t port = 9;

	// Correctly obtain the address of N0, the receiver
	Ipv4Address receiverAddress = interfaces.GetAddress(0);
	Address sinkAddress(InetSocketAddress(receiverAddress, port));

	Ptr<MyTcpApp> senderApp = CreateObject<MyTcpApp>();
	senderApp->Setup(nullptr, sinkAddress, 1024, DataRate("0.1Mbps")); // Configure your app
	terminals.Get(9)->AddApplication(senderApp); // Install the app on N9, the sender
	senderApp->SetStartTime(Seconds(0.1));
	senderApp->SetStopTime(Seconds(simLength));

	PacketSinkHelper sink("ns3::TcpSocketFactory", InetSocketAddress(Ipv4Address::GetAny(), port));
	ApplicationContainer sinkApp = sink.Install(terminals.Get(0)); // N0 as receiver
	sinkApp.Start(Seconds(0.0));
	sinkApp.Stop(Seconds(simLength));

    

    NS_LOG_INFO("Configure Tracing.");

    AsciiTraceHelper ascii;
    csma.EnableAsciiAll("Project");
    csma.EnablePcapAll("Project", false);


    // Run Simulation
    NS_LOG_INFO("Run Simulation.");
    float total = 0;
    
    Time totalRtt(0);
    uint32_t RttCount = 0;
    Time sessionRtt(0);
    uint32_t sessionRttCount = 0;
    Time avgRtt(0);
    Time lastTotalRtt(0);
    uint32_t lastRttCount = 0;
    
    uint32_t packetCount = 0;
    uint32_t totalPacketSize = 0;
    uint32_t sessionPacketCount = 0;
    uint32_t sessionPacketSize = 0;
    uint32_t lastPacketSize = 0;
    uint32_t lastPacketCount = 0;
    do
    {
        total++;
        Simulator::Stop(Seconds(1.0));
        Simulator::Run();
        
        
        totalRtt = senderApp->GetTotalRtt();
		RttCount = senderApp->GetRttCount();
		if(RttCount != lastRttCount){
		    sessionRtt = totalRtt - lastTotalRtt;
			sessionRttCount = RttCount - lastRttCount;
			avgRtt = sessionRtt / sessionRttCount;
		}
		lastTotalRtt = totalRtt;
		lastRttCount = RttCount;
		
		
		totalPacketSize = senderApp->GetTotalPacketSize();
		packetCount = senderApp->GetPacketCount();
		sessionPacketSize = totalPacketSize - lastPacketSize;
		sessionPacketCount = packetCount - lastPacketCount;
		lastPacketSize = totalPacketSize;
		lastPacketCount = packetCount;
			
		std::cout << "Bytes sent: " << sessionPacketSize << "  ||  Packets sent: " << sessionPacketCount << "  ||  Average RTT: " << avgRtt << std::endl;
		
		//std::cout << "Total RTT: " << totalRtt << std::endl;
		//std::cout << "RTT count: " << RttCount << std::endl;

    } while (total<20);

    
    Simulator::Destroy();

    NS_LOG_INFO("Done.");

    return 0;
}
