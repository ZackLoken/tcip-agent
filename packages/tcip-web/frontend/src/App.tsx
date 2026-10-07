import { ProjectPicker } from "@/components/ProjectPicker";
import { Toasts } from "@/components/Toasts";
import { ProjectShell } from "@/ProjectShell";
import { useStore } from "@/store";
import { selectAnnotatorNamed } from "@/store/slices/user";

/** The one admission point: the project shell (state socket, panel events, tab sync, session end,
 *  bars, rail and tabs) mounts only while the person's name is committed; until then App renders
 *  the project picker alone. */
function App() {
  const annotatorNamed = useStore(selectAnnotatorNamed);

  return (
    <div className="h-full flex flex-col bg-tcip-bg text-tcip-fg">
      {annotatorNamed ? <ProjectShell /> : <ProjectPicker />}
      <Toasts />
    </div>
  );
}

export default App;
