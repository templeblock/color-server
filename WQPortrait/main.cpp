#include "wqportrait.h"
#include <QtWidgets/QApplication>

int main(int argc, char *argv[])
{
	QApplication a(argc, argv);
	WQPortrait w;
	w.show();
	return a.exec();
}
