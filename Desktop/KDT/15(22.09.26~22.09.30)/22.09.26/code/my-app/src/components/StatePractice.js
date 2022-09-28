import { useState } from "react";

export default function StatePractice() {
  const [message, setMessage] = useState("");

  const onclickEnter = () => {
    setMessage("안녕하세요!");
  };

  const onclickLeave = () => {
    setMessage("안녕히 가세요!");
  };

  return (
    <div>
      <h1>{message}</h1>
      <button onclick={onclickEnter}>입장</button>
      <button onclick={onclickLeave}>퇴장</button>
    </div>
  );
}
