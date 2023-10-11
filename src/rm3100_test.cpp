extern "C"{
    #include "rm3100_spi_userspace.h"
}
#include <math.h>
#include <chrono>


uint64_t timeSinceEpochMillisec() {
  using namespace std::chrono;
  return duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
}

int main(){
    int ret;
    ret=start_mag();
    if(ret<0){
        printf("open mag failed");
        return 0;
    }
    ret=start_drdy();
    if(ret<0){
        printf("open drdy failed");
        end_mag();
        return 0;
    }
    int revid_id=get_revid_id();
    printf("REVID(correct ID is 22): %X\n",revid_id);
    uint16_t cycle_count=50;
    change_cycle_count(cycle_count);
    // set_continuous_measurement(true);
    // get_measurement_ready();
    set_continuous_measurement(false);
    // get_measurement();
    uint8_t tmrc_value=0x92;
    set_tmrc(tmrc_value);
    // struct Measurements res = single_measurement();
    // clear_drdy(drdy_);
    set_continuous_measurement(true);
    struct Measurements res;
    // struct Measurements res = get_measurement(drdy_,-1);
    // struct Measurements res = get_measurement_poll();
    // printf("x:%ld y:%ld pass:%d\n",res.x,res.y,res.ret);
    uint64_t start_ms=timeSinceEpochMillisec();

    int count=1000;
    for(int i=0; i<count;i++){
        res = get_measurement(800*1E6,true);
        if (res.ret!=1)break;
        // struct Measurements res = get_measurement_poll();
        // printf("x:%ld y:%ld z:%ld\n",res.x,res.y,res.z);
        printf("x:%ld y:%ld pass:%d\n",res.x,res.y,res.ret);
        printf("angle:%Lf\n",atan2((double)res.y,(double)res.x)+M_PIl);
    }
    uint64_t stop_ms=timeSinceEpochMillisec();
    printf("rate: %f Hz\n",1000*count/(float_t)(stop_ms-start_ms));

}
