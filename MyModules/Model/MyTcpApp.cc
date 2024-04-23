// MyTcpApp.cc
#include "MyTcpApp.h"
#include "ns3/simulator.h"
#include "ns3/socket.h"
#include "ns3/tcp-socket-factory.h"
#include "ns3/uinteger.h"
#include "ns3/log.h"
#include "ns3/inet-socket-address.h"
#include "ns3/packet.h"
#include "ns3/tcp-socket.h"


namespace ns3 {

NS_LOG_COMPONENT_DEFINE("MyTcpApp");
NS_OBJECT_ENSURE_REGISTERED(MyTcpApp);

TypeId MyTcpApp::GetTypeId(void)
{
    static TypeId tid = TypeId("ns3::MyTcpApp")
        .SetParent<Application>()
        .SetGroupName("Tutorial")
        .AddConstructor<MyTcpApp>();
    return tid;
}

MyTcpApp::MyTcpApp()
    : m_socket(0),
      m_peer(),
      m_packetSize(0),
      m_dataRate(0),
      m_sendEvent(),
      m_running(false),
      m_packetsSent(0),
      m_totalRtt(Seconds(0)), 
      m_totalPacketSize(0),
      m_packetCount(0) 
{
}


MyTcpApp::~MyTcpApp()
{
    m_socket = 0;
}

void MyTcpApp::Setup(Ptr<Socket> socket, Address address, uint32_t packetSize, DataRate dataRate)
{
    m_socket = socket;
    m_peer = address;
    m_packetSize = packetSize;
    m_dataRate = dataRate;
}

void MyTcpApp::StartApplication(void)
{
    m_running = true;
    m_packetsSent = 0;
    if (!m_socket)
    {
        m_socket = Socket::CreateSocket(GetNode(), TcpSocketFactory::GetTypeId());
        m_socket->Bind();
        m_socket->Connect(m_peer);

        // Cast the socket to TcpSocket to access the RTT trace source
        Ptr<TcpSocket> tcpSocket = DynamicCast<TcpSocket>(m_socket);
        if (tcpSocket)
        {
            tcpSocket->TraceConnectWithoutContext("RTT", MakeCallback(&MyTcpApp::RttChange, this));
        }
        else
        {
            NS_LOG_WARN("MyTcpApp: Socket created is not of type TcpSocket.");
        }
    }

    ScheduleTx();
}


void MyTcpApp::StopApplication(void)
{
    m_running = false;
    if (m_sendEvent.IsRunning())
    {
        Simulator::Cancel(m_sendEvent);
    }
    if (m_socket)
    {
        m_socket->Close();
        m_socket = 0;
    }
}

void MyTcpApp::ScheduleTx(void)
{
    if (m_running)
    {
        Time tNext(Seconds(m_packetSize * 8 / static_cast<double>(m_dataRate.GetBitRate())));
        m_sendEvent = Simulator::Schedule(tNext, &MyTcpApp::SendPacket, this);
    }
}

void MyTcpApp::SendPacket(void)
{
    Ptr<Packet> packet = Create<Packet>(m_packetSize);
    m_socket->Send(packet);
    m_totalPacketSize = m_totalPacketSize + m_packetSize;
    m_packetCount += 1;
    


    if (m_running)
    {
        ScheduleTx();
    }
}

void MyTcpApp::RttChange(Time oldRtt, Time newRtt)
{
    m_totalRtt = m_totalRtt + newRtt; // Update the last known RTT 
}




} // Namespace ns3
