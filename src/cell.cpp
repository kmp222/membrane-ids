#include "cell.hpp"

#include <functional>

Membrane* Cell::findMembrane(std::string ID) const {
    return findMembrane(rootMembrane, ID);
}

Membrane* Cell::findMembrane(Membrane* membrane, std::string ID) const {

    if (!membrane) return nullptr;
    if (membrane->ID == ID) return membrane;

    for (auto& child: membrane->subMembranes) {
        Membrane* result = findMembrane(child, ID);
        if (result) return result;
    }

    return nullptr;

}

void Cell::addMembrane(std::string parentID, std::string childID, MembraneType type) const {

    Membrane* parent = findMembrane(parentID);

    if (parent) {
        Membrane* child = new Membrane();
        child->ID = childID;
        child->type = type;
        child->parent = parent;
        parent->subMembranes.push_back(child);
    }
    
}

void Cell::generateRules() const {

    std::vector<Membrane*> allMembranes;

    std::function<void(Membrane*)> getAllMembranes = [&](Membrane* m) {
        if (m) {
            allMembranes.push_back(m);
            for (auto& child : m->subMembranes) {
                getAllMembranes(child);
            }
        }
    };

    getAllMembranes(rootMembrane);

    for (auto& m : allMembranes) {

        switch (m->type) {

        case MembraneType::Nucleus:

            m->rules.push_back(Rule({ m->parent->ID }, { m->parent->ID + "'" }, m->parent->ID));
            break;

        case MembraneType::Machine:

            m->rules.push_back(Rule({ m->ID + "'" }, { m->ID + "'" }, m->parent->ID));
            m->rules.push_back(Rule({ m->ID }, { m->ID + "'" }, m->subMembranes[0]->ID));
            break;

        case MembraneType::Router:

            m->rules.push_back(Rule({ m->ID }, { m->ID }, m->ID + "s"));

            if (m->parent != nullptr) {
                m->rules.push_back(Rule({ m->parent->ID, m->ID + "'" },
                    { m->parent->ID, m->ID + "'" }, m->parent->ID));
            }
            
            for (auto& child : m->subMembranes) {
                if (child->ID.back() != 's') {
                    m->rules.push_back(Rule({ child->ID, m->ID + "'" },
                        { child->ID, m->ID + "'" }, child->ID));
                }
            }

            break;

        case MembraneType::SrcCheck:

            for (auto& m2 : allMembranes) {

                if (m2->type == MembraneType::Machine) {
                    m->rules.push_back(Rule({ m2->ID + "'", m2->ID + "sc" },
                        { m2->ID + "''", m2->ID + "sc" }, m->subMembranes[0]->ID));

                    m->rules.push_back(Rule({ m2->ID + "''"},
                        { m2->ID + "'"}, m->parent->ID));
                }

            }

            break;

        case MembraneType::DstCheck:

            for (auto& m2 : allMembranes) {
                if (m2->type == MembraneType::Machine) {
                    m->rules.push_back(Rule({ m->parent->parent->ID, m2->ID + "dc" },
                        { m->parent->parent->ID + "'", m2->ID + "dc" }, m->parent->ID));
                }
            }

            break;

        }

    }

}

void Cell::print() const {
    std::function<void(const Membrane*, int)> printMembrane = [&](const Membrane* membrane, int level) {
        if (!membrane) return;

        // indentation for visual hierarchy
        std::string indentation(level * 2, ' ');
        std::cout << indentation << "Membrane ID: " << membrane->ID << ", Type: " << membrane->getType() << std::endl;

        // print objects in the membrane
        std::cout << indentation << "  Objects:" << std::endl;
        for (const auto& objectSet : membrane->objects) {
            std::cout << indentation << "    ";
            for (const auto& obj : objectSet) {
                std::cout << obj << " ";
            }
            std::cout << std::endl;
        }

        for (const auto& child : membrane->subMembranes) {
            printMembrane(child, level + 1);
        }
        };

    printMembrane(rootMembrane, 0);
}



