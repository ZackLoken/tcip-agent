/**
 * A staged agent request that follows the dataset selection until the breeder edits it.
 *
 * The selection resolves after first render, so the default text is rebuilt as it lands. Once the
 * text has been edited by hand, rebuilding it would throw that edit away, so it stops.
 */

import { useEffect, useState } from "react";

export interface EditableAgentRequest {
  request: string;
  setRequest: (text: string) => void;
}

export function useEditableAgentRequest(defaultRequest: string): EditableAgentRequest {
  const [request, setRequestText] = useState(defaultRequest);
  const [touched, setTouched] = useState(false);

  useEffect(() => {
    if (!touched) setRequestText(defaultRequest);
  }, [defaultRequest, touched]);

  return {
    request,
    setRequest: (text: string) => {
      setTouched(true);
      setRequestText(text);
    },
  };
}
