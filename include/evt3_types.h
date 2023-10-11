// -*-c++-*--------------------------------------------------------------------
// Copyright 2021 Bernd Pfrommer <bernd.pfrommer@gmail.com>
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef EVENT_CAMERA_CODECS__EVT3_TYPES_H_
#define EVENT_CAMERA_CODECS__EVT3_TYPES_H_

#include <stddef.h>
#include <stdint.h>

#include <fstream>
#include <iostream>

namespace event_camera_codecs
{
namespace evt3
{
enum Code {
  ADDR_Y = 0b0000,       // 0
  ADDR_X = 0b0010,       // 2
  VECT_BASE_X = 0b0011,  // 3
  VECT_12 = 0b0100,      // 4
  VECT_8 = 0b0101,       // 5
  TIME_LOW = 0b0110,     // 6
  CONTINUED_4 = 0b0111,  // 7
  TIME_HIGH = 0b1000,    // 8
  EXT_TRIGGER = 0b1010,  // 10
  OTHERS = 0b1110,       // 14
  CONTINUED_12 = 0b1111  // 15
};

enum SubType {
  MASTER_SYSTEM_TEMPERATURE = 0x000,
  MASTER_SYSTEM_VOLTAGE = 0x001,
  MASTER_SYSTEM_IN_EVENT_COUNT = 0x002,
  MASTER_SYSTEM_IN_EVENT_SEQ_ERROR = 0x003,
  MASTER_SYSTEM_IN_EVENT_TIME_ERROR = 0x004,
  MASTER_SYSTEM_OUT_EVENT_COUNT = 0x005,
  MASTER_SYSTEM_OUT_EVENT_SEQ_ERROR = 0x006,
  MASTER_SYSTEM_OUT_EVENT_TIME_ERROR = 0x007,
  MASTER_IN_TD_EVENT_COUNT = 0x014,
  MASTER_IN_APS_EVENT_COUNT = 0x015,
  MASTER_RATE_CONTROL_TD_EVENT_COUNT = 0x016,
  MASTER_RATE_CONTROL_APS_EVENT_COUNT = 0x017,
  MASTER_START_OF_FRAME = 0x018,
  MASTER_END_OF_FRAME = 0x019,
  MASTER_MIPI_PADDING = 0x01A,
  LOW_POWER_STATE_ENTRY_1 = 0x020,
  LOW_POWER_STATE_ENTRY_2 = 0x021,
  LOW_POWER_STATE_DEEP_EXIT = 0x022,
  END_OF_TEST_TASK = 0x0FD,
  USB_PACKET_INFO = 0x0FE,
  DUMMY = 0x0FF,
  MASTER_TL_DROP_EVENT = 0xED6,
  MASTER_TH_DROP_EVENT = 0xED8,
  MASTER_EVT_DROP_EVENT = 0xEDA
};

struct __attribute__((packed)) Event
{
  uint16_t rest : 12;
  uint16_t code : 4;
};

struct __attribute__((packed)) AddrY
{
  AddrY(uint16_t y_a, uint8_t s) : y(y_a), system_type(s), code(Code::ADDR_Y) {}
  uint16_t y : 11;
  uint16_t system_type : 1;
  uint16_t code : 4;
};

struct __attribute__((packed)) AddrX
{
  AddrX(uint16_t x_a, uint8_t p) : x(x_a), polarity(p), code(Code::ADDR_X) {}
  uint16_t x : 11;
  uint16_t polarity : 1;
  uint16_t code : 4;
};

struct __attribute__((packed)) TimeHigh
{
  TimeHigh(uint32_t ts_usec) : t(ts_usec), code(Code::TIME_HIGH) {}
  uint16_t t : 12;
  uint16_t code : 4;
};

struct __attribute__((packed)) TimeLow
{
  TimeLow(uint32_t ts_usec) : t(ts_usec & 0x00000FFF), code(Code::TIME_LOW) {}
  uint16_t t : 12;
  uint16_t code : 4;
};

struct __attribute__((packed)) Others
{
  Others(uint16_t s) : subtype(s), code(Code::OTHERS) {}
  uint16_t subtype : 12;
  uint16_t code : 4;
};

struct __attribute__((packed)) VectBaseX
{
  VectBaseX(uint16_t xa, uint16_t pa) : x(xa), pol(pa), code(Code::VECT_BASE_X) {}
  uint16_t x : 11;
  uint16_t pol : 1;
  uint16_t code : 4;
};

struct __attribute__((packed)) Vect8
{
  Vect8(uint16_t v) : valid(v), code(Code::VECT_8) {}
  uint16_t valid : 8;
  uint16_t unused : 4;
  uint16_t code : 4;
};

struct __attribute__((packed)) Vect12
{
  Vect12(uint16_t v) : valid(v), code(Code::VECT_12) {}
  uint16_t valid : 12;
  uint16_t code : 4;
};

struct __attribute__((packed)) ExtTrigger
{
  ExtTrigger(uint8_t e, uint8_t id_a) : edge(e), id(id_a), code(Code::EXT_TRIGGER) {}
  uint16_t edge : 1;
  uint16_t unused : 7;
  uint16_t id : 4;
  uint16_t code : 4;
};

inline std::string toString(const SubType s)
{
  switch (s) {
    case MASTER_SYSTEM_TEMPERATURE:
      return ("MASTER_SYSTEM_TEMPERATURE");
    case MASTER_SYSTEM_VOLTAGE:
      return ("MASTER_SYSTEM_VOLTAGE");
    case MASTER_SYSTEM_IN_EVENT_COUNT:
      return ("MASTER_SYSTEM_IN_EVENT_COUNT");
    case MASTER_SYSTEM_IN_EVENT_SEQ_ERROR:
      return ("MASTER_SYSTEM_IN_EVENT_SEQ_ERROR");
    case MASTER_SYSTEM_IN_EVENT_TIME_ERROR:
      return ("MASTER_SYSTEM_IN_EVENT_TIME_ERROR");
    case MASTER_SYSTEM_OUT_EVENT_COUNT:
      return ("MASTER_SYSTEM_OUT_EVENT_COUNT");
    case MASTER_SYSTEM_OUT_EVENT_SEQ_ERROR:
      return ("MASTER_SYSTEM_OUT_EVENT_SEQ_ERROR");
    case MASTER_SYSTEM_OUT_EVENT_TIME_ERROR:
      return ("MASTER_SYSTEM_OUT_EVENT_TIME_ERROR");
    case MASTER_IN_TD_EVENT_COUNT:
      return ("MASTER_IN_TD_EVENT_COUNT");
    case MASTER_IN_APS_EVENT_COUNT:
      return ("MASTER_IN_APS_EVENT_COUNT");
    case MASTER_RATE_CONTROL_TD_EVENT_COUNT:
      return ("MASTER_RATE_CONTROL_TD_EVENT_COUNT");
    case MASTER_RATE_CONTROL_APS_EVENT_COUNT:
      return ("MASTER_RATE_CONTROL_APS_EVENT_COUNT");
    case MASTER_START_OF_FRAME:
      return ("MASTER_START_OF_FRAME");
    case MASTER_END_OF_FRAME:
      return ("MASTER_END_OF_FRAME");
    case MASTER_MIPI_PADDING:
      return ("MASTER_MIPI_PADDING");
    case LOW_POWER_STATE_ENTRY_1:
      return ("LOW_POWER_STATE_ENTRY_1");
    case LOW_POWER_STATE_ENTRY_2:
      return ("LOW_POWER_STATE_ENTRY_2");
    case LOW_POWER_STATE_DEEP_EXIT:
      return ("LOW_POWER_STATE_DEEP_EXIT");
    case END_OF_TEST_TASK:
      return ("END_OF_TEST_TASK");
    case USB_PACKET_INFO:
      return ("USB_PACKET_INFO");
    case DUMMY:
      return ("DUMMY");
    case MASTER_TL_DROP_EVENT:
      return ("MASTER_TL_DROP_EVENT");
    case MASTER_TH_DROP_EVENT:
      return ("MASTER_TH_DROP_EVENT");
    case MASTER_EVT_DROP_EVENT:
      return ("MASTER_EVT_DROP_EVENT");
  }
  return ("UNKNOWN");
}
}  // end of namespace evt3
}  // namespace event_camera_codecs
#endif  // EVENT_CAMERA_CODECS__EVT3_TYPES_H_
