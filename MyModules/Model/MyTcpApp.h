// MyTcpApp.h
#ifndef MY_TCP_APP_H
#define MY_TCP_APP_H

#include "ns3/application.h"
#include "ns3/ptr.h"
#include "ns3/address.h"
#include "ns3/data-rate.h"
#include "ns3/event-id.h"
#include "ns3/traced-callback.h"

namespace ns3 {

class Socket;
class Packet;

class MyTcpApp : public Application 
{
public:
    static TypeId GetTypeId (void);
    MyTcpApp ();
    virtual ~MyTcpApp ();
    void Setup (Ptr<Socket> socket, Address address, uint32_t packetSize, DataRate dataRate);

    // RTT change callback method - ensure it's declared only once
    void RttChange(Time oldRtt, Time newRtt);
    
    Time GetTotalRtt(void) const { return m_totalRtt; } // Public getter for the total RTT
    uint32_t GetRxCount(void) const { return m_RxCount; }
    uint32_t GetTotalPacketSize(void) const { return m_totalPacketSize; }
    uint32_t GetPacketCount(void) const { return m_packetCount; }



protected:
    virtual void StartApplication (void);
    virtual void StopApplication (void);
    

private:
    void ScheduleTx (void);
    void SendPacket (void);

    Ptr<Socket> m_socket;
    Address m_peer;
    uint32_t m_packetSize;
    DataRate m_dataRate;
    EventId m_sendEvent;
    bool m_running;
    uint32_t m_packetsSent;

    void ConnectionSucceeded (Ptr<Socket> socket);
    void ConnectionFailed (Ptr<Socket> socket);
    
    Time m_totalRtt;
    uint32_t m_RxCount;
    uint32_t m_totalPacketSize;
    uint32_t m_packetCount;
};

} // namespace ns3

#endif // MY_TCP_APP_H
