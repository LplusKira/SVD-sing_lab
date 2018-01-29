#!/bin/bash

awk '{print $0" "int(rand() * 3);}' 0.edges.u2u > 0.edges.u2u.app
awk '{print $0" "int(rand() * 3);}' 107.edges.u2u > 107.edges.u2u.app
awk '{print $0" "int(rand() * 3);}' 1684.edges.u2u > 1684.edges.u2u.app
awk '{print $0" "int(rand() * 3);}' 1912.edges.u2u > 1912.edges.u2u.app
awk '{print $0" "int(rand() * 3);}' 3437.edges.u2u > 3437.edges.u2u.app
awk '{print $0" "int(rand() * 3);}' 348.edges.u2u > 348.edges.u2u.app
awk '{print $0" "int(rand() * 3);}' 3980.edges.u2u > 3980.edges.u2u.app
awk '{print $0" "int(rand() * 3);}' 414.edges.u2u > 414.edges.u2u.app
awk '{print $0" "int(rand() * 3);}' 686.edges.u2u > 686.edges.u2u.app
awk '{print $0" "int(rand() * 3);}' 698.edges.u2u > 698.edges.u2u.app
