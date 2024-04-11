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



Time RttCalc(Time totalRttCalc, int RttCountCalc, Time lastTotalRttCalc, int lastRttCountCalc){
    Time sessionRtt(0);
    uint32_t sessionRttCount = 0;
    if(RttCountCalc != lastRttCountCalc){
		    sessionRtt = totalRttCalc - lastTotalRttCalc;
			sessionRttCount = RttCountCalc - lastRttCountCalc;
			return(sessionRtt / sessionRttCount);
		}
    return Time(0);
}

void SetTcpCongestionControl(Ptr<Node> node, std::string tcpVariant) {
    // Lookup the TypeId for the desired TCP variant
    TypeId tid = TypeId::LookupByName(tcpVariant);
    // Create a configuration path for the specific node
    std::stringstream nodeId;
    nodeId << node->GetId();
    std::string specificNode = "/NodeList/" + nodeId.str() + "/$ns3::TcpL4Protocol/SocketType";
    // Set the TCP variant for the specific node
    Config::Set(specificNode, TypeIdValue(tid));
}



int main(int argc, char *argv[])
{
    RngSeedManager::SetSeed(1);
    RngSeedManager::SetRun(1);
    
    
    float simLength = 420.0;

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

    //Set global CCA
    //Config::SetDefault("ns3::TcpL4Protocol::SocketType", StringValue("ns3::TcpLinuxReno"));
    

    NodeContainer csmaSwitch;
    csmaSwitch.Create(2);

    NS_LOG_INFO("Build Topology");
    CsmaHelper csma;
    csma.SetChannelAttribute("DataRate", DataRateValue(DataRate("800kbps")));
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

    // Set TCP LinuxReno for node 9
    SetTcpCongestionControl(terminals.Get(9), "ns3::TcpLinuxReno");
    // Set TCP Cubic for node 8
    SetTcpCongestionControl(terminals.Get(8), "ns3::TcpCubic");

	// Correctly obtain the address of N0, the receiver
	Ipv4Address receiverAddress0 = interfaces.GetAddress(0);
	Address sinkAddress0(InetSocketAddress(receiverAddress0, port));

    // Correctly obtain the address of N1, the receiver
	Ipv4Address receiverAddress1 = interfaces.GetAddress(1);
	Address sinkAddress1(InetSocketAddress(receiverAddress1, port));

    // Set up N9 as sender app
	Ptr<MyTcpApp> senderApp9 = CreateObject<MyTcpApp>();
	senderApp9->Setup(nullptr, sinkAddress0, 1024, DataRate("4Mbps")); // Configure your app
	terminals.Get(9)->AddApplication(senderApp9); // Install the app on N9, the sender
	senderApp9->SetStartTime(Seconds(0.1));
	senderApp9->SetStopTime(Seconds(simLength));

    // Set up N8 as sender app
	/*
	Ptr<MyTcpApp> senderApp8 = CreateObject<MyTcpApp>();
	senderApp8->Setup(nullptr, sinkAddress1, 1024, DataRate("2Mbps")); // Configure your app
    terminals.Get(8)->AddApplication(senderApp8); // Install the app on N8, the sender
	senderApp8->SetStartTime(Seconds(200));
	senderApp8->SetStopTime(Seconds(simLength));
    */

    OnOffHelper onoff2 ("ns3::TcpSocketFactory", Address(InetSocketAddress(Ipv4Address("10.1.1.3"), port)));
    onoff2.SetConstantRate(DataRate("400kbps"));

    ApplicationContainer app2 = onoff2.Install(terminals.Get(7));
    app2.Start(Seconds(200));
    app2.Stop(Seconds(simLength));

    // Create packetsink helper
	PacketSinkHelper sink("ns3::TcpSocketFactory", InetSocketAddress(Ipv4Address::GetAny(), port));

    // Create sink apps
	ApplicationContainer sinkApp0 = sink.Install(terminals.Get(0)); // N0 as receiver
    ApplicationContainer sinkApp1 = sink.Install(terminals.Get(1)); // N1 as receiver

    // Start and stop sinkapps
	sinkApp0.Start(Seconds(0.0));
	sinkApp0.Stop(Seconds(simLength));

    /*
    sinkApp1.Start(Seconds(0.0));
	sinkApp1.Stop(Seconds(simLength));
    */

    

    NS_LOG_INFO("Configure Tracing.");

    AsciiTraceHelper ascii;
    csma.EnableAsciiAll("Project");
    csma.EnablePcapAll("Project", false);


    
    float total = 0;
    
    Time totalRtt(0);
    uint32_t RttCount = 0;
    Time avgRtt(0);
    
    Time lastTotalRtt9(0);
    uint32_t lastRttCount9 = 0;
    Time lastTotalRtt8(0);
    uint32_t lastRttCount8 = 0;
    uint32_t sessionRttCount = 0;

    uint32_t packetCount = 0;
    uint32_t totalPacketSize = 0;
    uint32_t sessionPacketCount = 0;
    uint32_t sessionPacketSize = 0;
    uint32_t lastPacketSize9 = 0;
    uint32_t lastPacketCount9 = 0;
    uint32_t lastPacketSize8 = 0;
    uint32_t lastPacketCount8 = 0;

    uint32_t totalRxBytes = 0;
    uint32_t sessionRxBytes = 0;
    uint32_t lastRxBytes = 0;

    //NS_LOG_INFO("Run Simulation.");
    std::cout << "Run Simulation" << std::endl;

    std::string command = "";

    // Read input from Python process
    while(command.empty()){
    	std::this_thread::sleep_for(std::chrono::milliseconds(5));
    	std::getline(std::cin, command);
    }
    
    do
    {
        total++;
        Simulator::Stop(Seconds(1.0));
        Simulator::Run();
        
        
        totalRtt = senderApp9->GetTotalRtt();
		RttCount = senderApp9->GetRttCount();
		avgRtt = RttCalc(totalRtt, RttCount, lastTotalRtt9, lastRttCount9);
        sessionRttCount = RttCount - lastRttCount9;
        lastTotalRtt9 = totalRtt;
	    lastRttCount9 = RttCount;
		
		totalPacketSize = senderApp9->GetTotalPacketSize();
        sessionPacketSize = totalPacketSize - lastPacketSize9;
        lastPacketSize9 = totalPacketSize;

		packetCount = senderApp9->GetPacketCount();
        sessionPacketCount = packetCount - lastPacketCount9;
        lastPacketCount9 = packetCount;
		
        
        totalRxBytes = DynamicCast<PacketSink>(sinkApp0.Get(0))->GetTotalRx();
        sessionRxBytes = totalRxBytes - lastRxBytes;
        lastRxBytes = totalRxBytes;

	    //std::cout << "N9 >> " << "Bytes sent: " << sessionPacketSize << "  ||  Tx: " << sessionPacketCount << "  ||  Rx: " << sessionRttCount << "  ||  Average RTT: " << avgRtt << std::endl;
	    //std::cout << sessionPacketCount << "," << sessionRttCount << ","  << sessionPacketSize << "," << avgRtt.GetMilliSeconds() << std::endl;
	    std::cout << sessionPacketCount << "," << sessionRttCount << ","  << sessionPacketSize << "," << sessionRxBytes << "," << avgRtt.GetMilliSeconds() << std::endl;
		
        /*
        totalRtt = senderApp8->GetTotalRtt();
		RttCount = senderApp8->GetRttCount();
		avgRtt = RttCalc(totalRtt, RttCount, lastTotalRtt8, lastRttCount8);
        sessionRttCount = RttCount - lastRttCount8;
        lastTotalRtt8 = totalRtt;
	    lastRttCount8 = RttCount;
		
		
		totalPacketSize = senderApp8->GetTotalPacketSize();
        sessionPacketSize = totalPacketSize - lastPacketSize8;
        lastPacketSize8 = totalPacketSize;

		packetCount = senderApp8->GetPacketCount();
        sessionPacketCount = packetCount - lastPacketCount8;
        lastPacketCount8 = packetCount;
		
	    std::cout << "N8 >> " << "Bytes sent: " << sessionPacketSize << "  ||  Tx: " << sessionPacketCount << "  ||  Rx: " << sessionRttCount << "  ||  Average RTT: " << avgRtt << std::endl;

        std::cout << std::endl;
        */
		//std::cout << "Total RTT: " << totalRtt << std::endl;
		//std::cout << "RTT count: " << RttCount << std::endl;

        command = "";
	    while(command.empty()){
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            std::getline(std::cin, command);
    	}

    } while (total<420);

    
    Simulator::Destroy();

    NS_LOG_INFO("Done.");

    return 0;
}