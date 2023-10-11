extern "C"{
    #include "rm3100_spi_userspace.h"
}
#include <math.h>
#include <chrono>
#include <iostream>
#include <linux/gpio.h>
#include <poll.h>
#define POLL_GPIO POLLPRI | POLLERR

uint64_t timeSinceEpochMillisec() {
  using namespace std::chrono;
  return duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
}

 
#define SYSFS_GPIO_DIR "/sys/class/gpio"
#define POLL_TIMEOUT (3 * 1000) /* 3 seconds */
#define MAX_BUF 64
/****************************************************************
 * gpio_set_edge
 ****************************************************************/

// int gpio_set_edge()
// {
// 	int fd, len;
// 	char buf[MAX_BUF];

// 	len = snprintf(buf, sizeof(buf), SYSFS_GPIO_DIR "/PN.01/edge");
 
// 	fd = open(buf, O_WRONLY);
// 	if (fd < 0) {
// 		perror("gpio/set-edge");
// 		return fd;
// 	}
 
// 	write(fd, "rising", 6); 
// 	close(fd);
// 	return 0;
// }

// int main(){

//     int fd = open("/sys/class/gpio/export", O_WRONLY);
//     if (fd == -1) {
//         perror("Unable to open /sys/class/gpio/export");
//         exit(1);
//     }

//     // if (write(fd, "433", 3) != 3) {
//     //     perror("Error writing to /sys/class/gpio/export");
//     //     exit(1);
//     // }

//     close(fd);
//     // Set the pin to be an input by writing "in" to /sys/class/gpio/PN.01/direction

//     fd = open("/sys/class/gpio/PN.01/direction", O_WRONLY);
//     if (fd == -1) {
//         perror("Unable to open /sys/class/gpio/PN.01/direction");
//         exit(1);
//     }

//     if (write(fd, "in", 3) != 3) {
//         perror("Error writing to /sys/class/gpio/PN.01/direction");
//         exit(1);
//     }

//     close(fd);

//     //test read

//     gpio_set_edge();
//     fd = open("/sys/class/gpio/PN.01/value", O_RDONLY);
//     if (fd == -1) {
//         perror("Unable to open /sys/class/gpio/PN.01/value");
//         exit(1);
//     }

//     struct pollfd fdset[2];
//     struct pollfd pollfd_1;
// 	int nfds = 2;
// 	int gpio_fd, timeout, rc;
//     timeout=-1;
// 	uint8_t *buf[MAX_BUF];
// 	unsigned int gpio=433;
// 	int len;

// 	while (1) {
// 		memset((void*)fdset, 0, sizeof(fdset));
// 		memset((void*)&pollfd_1, 0, sizeof(pollfd));

// 		// fdset[0].fd = STDIN_FILENO;
// 		// fdset[0].events = POLLIN;
      
// 		// fdset[1].fd = fd;
// 		// fdset[1].events = POLLPRI;
//         lseek(fd, 0, SEEK_SET);    /* consume any prior interrupt */
//         read(fd, buf, sizeof buf);
//         pollfd_1.fd=fd;
//         pollfd_1.events=POLLPRI;
//         pollfd_1.revents=0;
// 		// rc = poll(fdset, nfds, timeout); 
// 		rc = poll(&pollfd_1, 1, -1);      

// 		if (rc < 0) {
// 			printf("\npoll() failed!\n");
// 			return -1;
// 		}
      
// 		if (rc == 0) {
// 			printf(".");
// 		}
//         else{
//             printf("%d",rc);
//         }
            
// 		if (fdset[1].revents & POLL_GPIO) {
// 			// len = read(fdset[1].fd, buf, MAX_BUF);
// 			len = read(pollfd_1.fd, buf, MAX_BUF);
// 			printf("\npoll() GPIO %d interrupt occurred\n", gpio);
// 		}

// 		if (fdset[0].revents & POLLIN) {
// 			(void)read(fdset[0].fd, buf, 1);
// 			printf("\npoll() stdin read 0x%2.2X\n", (u_int32_t)*buf[0]);
// 		}

// 		fflush(stdout);
// 	}
//     close(fd);

//     // Unexport the pin by writing to /sys/class/gpio/unexport

//     fd = open("/sys/class/gpio/unexport", O_WRONLY);
//     if (fd == -1) {
//         perror("Unable to open /sys/class/gpio/unexport");
//         exit(1);
//     }

//     if (write(fd, "433", 2) != 2) {
//         perror("Error writing to /sys/class/gpio/unexport");
//         exit(1);
//     }

//     close(fd);
//     fd = start_mag();
//     int res=get_measurement_ready(fd);
// 	uint8_t rx[2] = {0, }; 
// 	read_addr(fd,0xA4,1,rx);
//     get_measurement_poll(fd);
//     if(res==1){
//         printf("drdy on\n");
//     }
//     else{
//         printf("drdy off\n");
//     }
//     end_mag(fd);
// }
