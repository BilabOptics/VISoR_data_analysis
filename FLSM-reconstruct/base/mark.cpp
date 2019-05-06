#include "mark.h"

using namespace flsm;

mark::mark()
{

}

mark::mark(Structure str, int group_)
{
    center = str.center();
    type = mktype_default;
    group = group_;
}
