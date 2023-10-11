// include API header for the new interface

#include <linux/gpio.h>

 

#include <unistd.h>

#include <fcntl.h>

#include <string.h>

#include <errno.h>

#include <sys/ioctl.h>

#include <stdint.h>

#include <stdlib.h>

 

#define DEV_NAME "/dev/gpiochip0"

 

int fd, ret;

fd = open(DEV_NAME, O_RDONLY);

if (fd < 0)

{

    printf("Unabled to open %%s: %%s", dev_name, strerror(errno));

    return;

}

/*

    control GPIO here, such as:

    - configure

    - read

    - write

    - polling

*/

(void)close(fd);

struct gpiochip_info info;

struct gpioline_info line_info;

int fd, ret;

// open the device

fd = open(DEV_NAME, O_RDONLY);

if (fd < 0)

{

    printf("Unabled to open %%s: %%s", dev_name, strerror(errno));

    return;

}

 

// Query GPIO chip information

ret = ioctl(fd, GPIO_GET_CHIPINFO_IOCTL, &info);

if (ret == -1)

{

    printf("Unable to get chip info from ioctl: %%s", strerror(errno));

    close(fd);

    return;

}

printf("Chip name: %%s\n", info.name);

printf("Chip label: %%s\n", info.label);

printf("Number of lines: %%d\n", info.lines);
 