#include <QApplication>
#include "qt5mainwindow.h"

int main(int argc, char** argv)
{
    QApplication app(argc,argv);
    QApplication::setApplicationName("LibNano");
    QApplication::setOrganizationName("LibNano");

    MainWindow wapp;
    wapp.setWindowTitle(QApplication::applicationName());
    wapp.show();

    return QApplication::exec();
}
