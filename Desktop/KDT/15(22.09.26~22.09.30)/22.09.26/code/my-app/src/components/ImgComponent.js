import React, { Component } from "react";
import monkeyImg from "../capuchine.jpg";

class ImgComponent extends Component {
  render() {
    return (
      <div>
        <img src={monkeyImg} alt="검은 머리 카푸친 원숭이" />
      </div>
    );
  }
}

/* function ImgComponent() {
  return <img src={monkeyImg} alt="검은 머리 카푸친 원숭이" />;
} */
export default ImgComponent;
