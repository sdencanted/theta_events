#!/bin/bash
for i in {316..600}
do
    PIN=$i
    
    sudo sh -c "echo $PIN > /sys/class/gpio/unexport"
done