extern "C"{
    #include "rm3100_spi_userspace.h"
}
#include <math.h>
#include <chrono>
#include <iostream>

uint64_t timeSinceEpochMillisec() {
  using namespace std::chrono;
  return duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
}

int main(){
    start_mag();
    int res=get_measurement_ready();
    // set_continuous_measurement(,false);
    usleep(10000);
    get_measurement_ready();
	uint8_t rx[2] = {0, }; 
	read_addr(0xA4,1,rx);
    // printf("drdy off\n");
    end_mag();

}
