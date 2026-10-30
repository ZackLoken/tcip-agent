import type { terminalApi } from "@/api/terminal";
import type { AuditGapRefusal, TerminalEnd, TerminalStatus } from "@/api/types.generated";
import type { Refusal } from "@/test/refusals";
import produced from "@/test/terminal.json";

type Launch = Awaited<ReturnType<typeof terminalApi.createSession>>;

/**
 * The agent terminal's route answers `tools/generate_terminal_fixture.py` received: the hand-off
 * of a session for no project, which ended it at once with the request its agent was never given
 * (`handed_off_at_once`); the `status` of every row available; a `launch` its end answered with
 * nothing (`ended`), a `relaunch` whose end answered its three requests (`ended_undelivered`),
 * and the end and hand-off of a session no longer held refused (`end_unknown`,
 * `hand_off_unknown`); a session's hand-off underway (`handed_off`); an end whose line could not be
 * written (`end_audit_gap`) and an end whose process survived (`end_survived`); and the status
 * with the third row's executable absent (`status_mixed`) and every row's (`status_absent`).
 */
// A JSON import reads each literal as a string, which the literal unions admit only through
// `unknown`.
export const TERMINAL_ANSWERS = produced as unknown as {
  handed_off_at_once: TerminalEnd;
  status: TerminalStatus;
  launch: Launch;
  ended: TerminalEnd;
  relaunch: Launch;
  ended_undelivered: TerminalEnd;
  end_unknown: Refusal<string>;
  hand_off_unknown: Refusal<string>;
  handed_off: null;
  end_audit_gap: Refusal<AuditGapRefusal & { committed: TerminalEnd }>;
  end_survived: Refusal<string>;
  status_mixed: TerminalStatus;
  status_absent: TerminalStatus;
};
