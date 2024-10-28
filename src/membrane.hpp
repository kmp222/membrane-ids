#ifndef MEMBRANE_HPP
#define MEMBRANE_HPP

#include <vector>
#include <string>
#include <memory>

#include "rule.hpp"

/*  MEMBRANES ID LEGEND:
    x -> machine / router
    xn -> machine nucleus
    xs -> router source check
    xd -> router destination check

    STRING-OBJECTS ALPHABET:
    x -> label
    x' -> label'
    x'' -> label''
    xsc -> x source catalyst
    xdc -> x destination catalyst
    xdst -> x destination
    xsrc -> x source

    A membrane contains a multiset of string-objects (e.g. of string-object: 'x y z b c xsrc xdst') to define packets.
    Catalysts are seen as atomic objects.    */
    
enum class MembraneType {
    Router,
    Machine,
    Nucleus,
    SrcCheck,
    DstCheck
};

struct Membrane {

    std::string ID;
    std::vector<std::vector<std::string>> objects;
    std::vector<Membrane*> subMembranes;
    std::vector<Rule> rules;
    Membrane* parent;
    MembraneType type;

    Membrane() : parent(nullptr), type(MembraneType::Nucleus) {};

    Membrane(const std::string& ID, MembraneType type, Membrane* parent = nullptr)
        : ID(ID), type(type), parent(parent) {}

    ~Membrane() {
        for (auto subMembrane : subMembranes) {
            delete subMembrane;
        }
    }

    std::string getType() const {
        switch (type) {
        case MembraneType::Router: return "Router";
        case MembraneType::Machine: return "Machine";
        case MembraneType::Nucleus: return "Nucleus";
        case MembraneType::SrcCheck: return "SrcCheck";
        case MembraneType::DstCheck: return "DstCheck";
        default: return "Unknown";
        }
    }

};

#endif