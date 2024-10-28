#ifndef CELL_HPP
#define CELL_HPP

#include "membrane.hpp"

struct Cell {

    Membrane* rootMembrane;

    Membrane* findMembrane(std::string ID) const;

    Membrane* findMembrane(Membrane* membrane, std::string ID) const;

    void addMembrane(std::string parentID, std::string childID, MembraneType type) const;

    void generateRules() const;

    void print() const;

};

#endif