#!/bin/bash
exit 0
#sh /path/to/sh/file/sh   gpu   data_fold
sh /home/duenias/PycharmProjects/HyperNetworks/final_experiments/Film.sh
sh /home/duenias/PycharmProjects/HyperNetworks/final_experiments/TabularAsHyper.sh
sh /home/duenias/PycharmProjects/HyperNetworks/final_experiments/DAFT.sh
sh /home/duenias/PycharmProjects/HyperNetworks/final_experiments/ImageAsHyper.sh

sh /home/duenias/PycharmProjects/HyperNetworks/final_experiments/TabularAsHyper.sh 2 0
sh /home/duenias/PycharmProjects/HyperNetworks/final_experiments/TabularAsHyper.sh 2 1
sh /home/duenias/PycharmProjects/HyperNetworks/final_experiments/TabularAsHyper.sh 1 2
sh /home/duenias/PycharmProjects/HyperNetworks/final_experiments/TabularAsHyper.sh 1 3

sh /home/duenias/PycharmProjects/HyperNetworks/final_experiments/TabularAsHyper1.sh 2 0
sh /home/duenias/PycharmProjects/HyperNetworks/final_experiments/TabularAsHyper1.sh 2 1
sh /home/duenias/PycharmProjects/HyperNetworks/final_experiments/TabularAsHyper1.sh 1 2
sh /home/duenias/PycharmProjects/HyperNetworks/final_experiments/TabularAsHyper1.sh 1 3
