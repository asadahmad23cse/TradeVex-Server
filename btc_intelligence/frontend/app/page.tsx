import LiveCandleChart from "../components/LiveCandleChart";
import SignalPanel from "../components/SignalPanel";

export default function HomePage() {
  return (
    <main className="mainGrid">
      <SignalPanel />
      <LiveCandleChart />
    </main>
  );
}
