#include <iostream>
#include <fstream>
#include <chrono>  // Include for time-related functionality
#include <numeric>
#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/applications-module.h"
#include "ns3/bridge-module.h"
#include "ns3/csma-module.h"
#include "ns3/internet-module.h"
#include <vector>
#include "ns3/tcp-socket.h"



using namespace ns3;

// Settings
float g_simLength = 420.0;
std::string g_throughput =       "120kBps";
std::string g_agentSendRate =    "200kBps";
std::string g_client1SendRate =   "60kBps";
std::string g_client2SendRate =   "30kBps";
u_int32_t g_packetSize = 512;


NS_LOG_COMPONENT_DEFINE("Project");
int g_ackReceived = 0;
int g_ackReceivedCubic = 0;
uint32_t g_packetCount = 0;
uint32_t g_bytesSent = 0;
Time g_totalRtt(0);

std::vector<double> g_RttList;

// (DataRate(g_throughput).GetBitRate())

uint32_t offset = 2;


void resetVals()
{
    g_packetCount = 0;
    g_bytesSent = 0;
    g_ackReceived = 0;
    g_ackReceivedCubic = 0;
    g_totalRtt = Seconds(0.0);
}

Time RttCalc(Time totalRttCalc, int RttCountCalc, Time lastTotalRttCalc){
    Time sessionRtt(0);
    if(RttCountCalc != 0){
		    sessionRtt = totalRttCalc - lastTotalRttCalc;
			return(sessionRtt / RttCountCalc);
		}
    return Time(0);
}

void TxPacketsTrace(Ptr<const Packet> packet)
{
    g_packetCount++;
    g_bytesSent += packet->GetSize(); // Accumulate the size of each sent packet
    //std::cout << packet->GetSize() << std::endl;
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

void RttChange(Time oldRtt, Time newRtt)
{
    g_totalRtt = g_totalRtt + newRtt; // Update the last known RTT 
    g_RttList.push_back(newRtt.GetMilliSeconds());

}

void RxPacketsTrace(Ptr<const Packet> packet, const Address &addr)
{
    g_ackReceived++;
}

void RxPacketsTraceCubic(Ptr<const Packet> packet, const Address &addr)
{
    g_ackReceivedCubic++;
}

void SetupRttTrace(Ptr<Application> app, const std::string& traceSource, Callback<void, Time, Time> callback) {
    Ptr<OnOffApplication> onoffApp = DynamicCast<OnOffApplication>(app);
    if (onoffApp) {
        Ptr<Socket> tcpSocket = onoffApp->GetSocket();
        if (tcpSocket) {
            tcpSocket->TraceConnectWithoutContext(traceSource, callback);
        } else {
            NS_LOG_UNCOND("Socket not initialized!");
        }
    } else {
        NS_LOG_UNCOND("Failed to cast to OnOffApplication!");
    }
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

    // Record start time using C++ chrono
    auto startTime = std::chrono::high_resolution_clock::now();

    NS_LOG_INFO("Create nodes.");
    NodeContainer terminals;
    terminals.Create(10);

    NodeContainer csmaSwitch;
    csmaSwitch.Create(2);

    NS_LOG_INFO("Build Topology");
    CsmaHelper csma;
    csma.SetChannelAttribute("DataRate", DataRateValue(DataRate(g_throughput)));
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

    OnOffHelper onoff0("ns3::TcpSocketFactory", Address(InetSocketAddress(Ipv4Address("10.1.1.1"), port)));
    onoff0.SetConstantRate(DataRate(g_agentSendRate),g_packetSize);

    OnOffHelper onoff1("ns3::TcpSocketFactory", Address(InetSocketAddress(Ipv4Address("10.1.1.2"), port)));
    onoff1.SetConstantRate(DataRate(g_throughput),g_packetSize);

    ApplicationContainer app1 = onoff0.Install(terminals.Get(9));
    app1.Get(0)->TraceConnectWithoutContext("Tx", MakeCallback(&TxPacketsTrace));
    
    app1.Start(Seconds(0));
    app1.Stop(Seconds(g_simLength));

    // Schedule RTT trace setup to occur shortly after the application starts
    Simulator::Schedule(Seconds(0.001), &SetupRttTrace, app1.Get(0), "RTT", MakeCallback(&RttChange));

    ApplicationContainer app2 = onoff1.Install(terminals.Get(8));
    
    app2.Start(Seconds(0));
    app2.Stop(Seconds(g_simLength));

    
    // Create packetsink helper
	PacketSinkHelper sink("ns3::TcpSocketFactory", InetSocketAddress(Ipv4Address::GetAny(), port));

    // Create sink apps
	ApplicationContainer sinkApp0 = sink.Install(terminals.Get(0)); // N0 as receiver
    sinkApp0.Get(0)->TraceConnectWithoutContext("Rx", MakeCallback(&RxPacketsTrace));

	ApplicationContainer sinkApp1 = sink.Install(terminals.Get(1)); // N1 as receiver
    sinkApp1.Get(0)->TraceConnectWithoutContext("Rx", MakeCallback(&RxPacketsTraceCubic));

    PacketSinkHelper udpSinkHelper("ns3::UdpSocketFactory", InetSocketAddress(Ipv4Address::GetAny(), port));
    ApplicationContainer udpSinkApp1 = udpSinkHelper.Install(terminals.Get(2));  
    ApplicationContainer udpSinkApp2 = udpSinkHelper.Install(terminals.Get(3));  

    // Start and stop sinkapps
	sinkApp0.Start(Seconds(0.0));
	sinkApp0.Stop (Seconds(g_simLength));

    sinkApp1.Start(Seconds(0.0));
	sinkApp1.Stop (Seconds(g_simLength));

    udpSinkApp1.Start(Seconds(0.0));
    udpSinkApp1.Stop (Seconds(g_simLength));

    udpSinkApp2.Start(Seconds(0.0));
    udpSinkApp2.Stop (Seconds(g_simLength));

    
    // ----------------------------------------------------------------------------------



    OnOffHelper onoff2 ("ns3::UdpSocketFactory", Address(InetSocketAddress(Ipv4Address("10.1.1.3"), port)));
    onoff2.SetConstantRate(DataRate(g_client1SendRate));

    ApplicationContainer app4 = onoff2.Install(terminals.Get(7));
    
    OnOffHelper onoff3 ("ns3::UdpSocketFactory", Address(InetSocketAddress(Ipv4Address("10.1.1.4"), port)));
    onoff3.SetConstantRate(DataRate(g_client2SendRate));

    ApplicationContainer app3 = onoff3.Install(terminals.Get(6));

    

    // 50%
    app4.Start(Seconds(offset + 130));
    app4.Stop( Seconds(g_simLength));
    






    NS_LOG_INFO("Configure Tracing.");

    AsciiTraceHelper ascii;
    csma.EnableAsciiAll("Project");
    csma.EnablePcapAll("Project", false);


    
    float total = 0;
    
    uint32_t RttCount = 0;
    double avgRtt = 0;


    //NS_LOG_INFO("Run Simulation.");
    std::cout << "Run Simulation" << std::endl;
    Simulator::Stop(Seconds(1.0));
    Simulator::Run();

    std::string command = "";

    // Read input from Python process
    while(command.empty()){
    	std::this_thread::sleep_for(std::chrono::milliseconds(5));
    	std::getline(std::cin, command);
    }

    do
    {
        total++;
        resetVals();
        g_RttList.clear();
        Simulator::Stop(Seconds(1.0));
        Simulator::Run();
        
		if(g_ackReceived!=0){
            avgRtt = g_totalRtt.GetMilliSeconds()/ (double) g_ackReceived;
        }

        double sq_sum = 0;
        if (g_ackReceived != 0){
            for (int ii = 0; ii < g_RttList.size(); ii++){
                sq_sum += std::pow(g_RttList[ii] - avgRtt, 2);

            }
        }
        double rttDev = 0;
        if (g_ackReceived != 0) {
            rttDev = std::sqrt(sq_sum / g_ackReceived);
        }
        

	    std::cout << g_packetCount << "," << g_ackReceived << ","  << g_bytesSent << "," << g_ackReceived * g_packetSize << "," << avgRtt << "," << rttDev << std::endl;
    	
        command = "";
	    while(command.empty()){
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            std::getline(std::cin, command);
    	}

	    std::cout << g_ackReceivedCubic * g_packetSize  << std::endl;

        command = "";
	    while(command.empty()){
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            std::getline(std::cin, command);
    	}
        
    } while (total<g_simLength);

    
    Simulator::Destroy();

    NS_LOG_INFO("Done.");

    return 0;
}
