#!/bin/bash
exit 0
#sh /path/to/sh/file/sh   gpu   data_fold
sh /home/duenias/PycharmProjects/HyperNetworks/final_experiments/Film.sh
sh /home/duenias/PycharmProjects/HyperNetworks/final_experiments/TabularAsHyper.sh
sh /home/duenias/PycharmProjects/HyperNetworks/final_experiments/DAFT.sh
sh /home/duenias/PycharmProjects/HyperNetworks/final_experiments/ImageAsHyper.sh

sh /home/duenias/PycharmProjects/HyperNetworks/final_experiments/TabularAsHyper.sh 0 0
sh /home/duenias/PycharmProjects/HyperNetworks/final_experiments/TabularAsHyper.sh 0 1
sh /home/duenias/PycharmProjects/HyperNetworks/final_experiments/TabularAsHyper.sh 2 2
sh /home/duenias/PycharmProjects/HyperNetworks/final_experiments/TabularAsHyper.sh 2 3

