cmake_minimum_required(VERSION 3.5)
project(thorvald_model)

find_package(ament_cmake REQUIRED)

# Instalar tudo que não é código: URDFs, meshes, configs, launch
install(DIRECTORY
  launch
  meshes
  urdf
  config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
