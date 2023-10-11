#!/bin/bash
for i in {433,450}
do
    PIN=$i
    
    sudo sh -c "echo $PIN > /sys/class/gpio/export"
    PINNO=$(ls /sys/class/gpio | grep P)
    if [ "$PINNO" !=  "" ];
    then
        
        echo $PINNO
        echo $i

        cat /sys/class/gpio/$PINNO/direction > /tmp/direction
        cat /tmp/direction
        sudo sh -c "echo in > /sys/class/gpio/$PINNO/direction"
        
        gpsum=0
        for i in {0..10}
        do
            build/drdy_on
            sleep 0.1
            gpval=$(cat /sys/class/gpio/$PINNO/value)
            gpsum=$(($gpsum + $gpval))
            # build/drdy_off
            sleep 0.1
            gpval=$(cat /sys/class/gpio/$PINNO/value)
            gpsum=$(($gpsum + $gpval))
            
        done
        echo $gpsum
        sudo sh -c "cat /tmp/direction > /sys/class/gpio/$PINNO/direction"
        sudo sh -c "echo $PIN > /sys/class/gpio/unexport"
    else
        echo $i does not exist
    fi
done