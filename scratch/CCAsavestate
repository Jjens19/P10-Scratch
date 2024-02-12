#include "tcp-linux-reno.h"
#include "ns3/log.h"
#include "ns3/simulator.h"

namespace ns3
{

NS_LOG_COMPONENT_DEFINE("TcpLinuxReno");
NS_OBJECT_ENSURE_REGISTERED(TcpLinuxReno);

TypeId
TcpLinuxReno::GetTypeId()
{
    static TypeId tid = TypeId("ns3::TcpLinuxReno")
                            .SetParent<TcpCongestionOps>()
                            .SetGroupName("Internet")
                            .AddConstructor<TcpLinuxReno>();
    return tid;
}

TcpLinuxReno::TcpLinuxReno()
    : TcpCongestionOps()
{
    NS_LOG_FUNCTION(this);
}

TcpLinuxReno::TcpLinuxReno(const TcpLinuxReno& sock)
    : TcpCongestionOps(sock)
{
    NS_LOG_FUNCTION(this);
}

TcpLinuxReno::~TcpLinuxReno()
{
}

uint32_t rate() {
	//Open file
	std::ifstream inputFile("/home/jakob/ns-allinone-3.40/ns-3.40/scratch/rate.txt");
	
	// Check if the file is open
    if (!inputFile.is_open()) {
        std::cerr << "Error opening the file." << std::endl;
        return 1; // Return an error code
    }
    
    int increase;
    inputFile >> increase;

	// Check if the read operation was successful
    if (!inputFile) {
        std::cerr << "Error reading from the file." << std::endl;
        return 1; // Return an error code
    }
    
    // Close the file
    inputFile.close();
    
    return increase;
}

uint32_t
TcpLinuxReno::SlowStart(Ptr<TcpSocketState> tcb, uint32_t segmentsAcked)
{
    NS_LOG_FUNCTION(this << tcb << segmentsAcked);
    
    // Set congestion window equal to rate
    tcb->m_cWnd = rate();
    
    if (tcb->m_cWnd < 1)
    {
        tcb->m_cWnd = 1; // Ensure the cwnd is at least 1
    }

    // Return the number of segments acknowledged
    return segmentsAcked;
}

void
TcpLinuxReno::CongestionAvoidance(Ptr<TcpSocketState> tcb, uint32_t segmentsAcked)
{
    NS_LOG_FUNCTION(this << tcb << segmentsAcked);
    
    // Set congestion window equal to rate
    tcb->m_cWnd = rate();
    if (tcb->m_cWnd < 1)
    {
        tcb->m_cWnd = 1; // Ensure the cwnd is at least 1
    }
}

void
TcpLinuxReno::IncreaseWindow(Ptr<TcpSocketState> tcb, uint32_t segmentsAcked)
{
    NS_LOG_FUNCTION(this << tcb << segmentsAcked);

    // Check if in slow start or congestion avoidance
    if (tcb->m_cWnd < tcb->m_ssThresh)
    {
        NS_LOG_DEBUG("In slow start, m_cWnd " << tcb->m_cWnd << " m_ssThresh " << tcb->m_ssThresh);
        segmentsAcked = SlowStart(tcb, segmentsAcked);
    }
    else
    {
        NS_LOG_DEBUG("In cong. avoidance, m_cWnd " << tcb->m_cWnd << " m_ssThresh "
                                                   << tcb->m_ssThresh);
        CongestionAvoidance(tcb, segmentsAcked);
    }
}

std::string
TcpLinuxReno::GetName() const
{
    return "TcpLinuxReno";
}

uint32_t
TcpLinuxReno::GetSsThresh(Ptr<const TcpSocketState> state, uint32_t bytesInFlight)
{
    NS_LOG_FUNCTION(this << state << bytesInFlight);

    // Decision making always returns a 1 
    return 1;
}

Ptr<TcpCongestionOps>
TcpLinuxReno::Fork()
{
    return CopyObject<TcpLinuxReno>(this);
}

} // namespace ns3

