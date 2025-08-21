#ifndef DFROBOT_QMC5883L_H
#define DFROBOT_QMC5883L_H

#if ARDUINO >= 100
#include "Arduino.h"
#else
#include "WProgram.h"
#endif

#define QMC5883L_ADDRESS              (0x0D)

#define QMC5883L_REG_OUT_X_M          (0x01)
#define QMC5883L_REG_OUT_X_L          (0x00)
#define QMC5883L_REG_OUT_Z_M          (0x05)
#define QMC5883L_REG_OUT_Z_L          (0x04)
#define QMC5883L_REG_OUT_Y_M          (0x03)
#define QMC5883L_REG_OUT_Y_L          (0x02)
#define QMC5883L_REG_STATUS           (0x06)
#define QMC5883L_REG_CONTROL_1        (0x09)
#define QMC5883L_REG_CONTROL_2        (0x0A)
#define QMC5883L_REG_IDENT_B          (0x0B)
#define QMC5883L_REG_IDENT_C          (0x0C)
#define QMC5883L_REG_IDENT_D          (0x0D)

typedef enum
{
  QMC5883L_DATARATE_10HZ        = 0b00,
  QMC5883L_DATARATE_50HZ        = 0b01,
  QMC5883L_DATARATE_100HZ       = 0b10,
  QMC5883L_DATARATE_200HZ       = 0b11
} qmc5883l_dataRate_t;

typedef enum
{
  QMC5883L_RANGE_2GA     = 0b00,
  QMC5883L_RANGE_8GA     = 0b01
} qmc5883l_range_t;

typedef enum
{
  QMC5883L_SINGLE        = 0b00,
  QMC5883L_CONTINOUS     = 0b01
} qmc5883l_mode_t;

typedef enum
{
  QMC5883L_OSR_64        = 0b11,
  QMC5883L_OSR_128       = 0b10,
  QMC5883L_OSR_256       = 0b01,
  QMC5883L_OSR_512       = 0b00,
} qmc5883l_osr_t;

#ifndef VECTOR_STRUCT_H
#define VECTOR_STRUCT_H
struct Vector
{
  int16_t XAxis;
  int16_t YAxis;
  int16_t ZAxis;
};
#endif

class QMC5883L
{
public:
 
  bool begin(void);

  Vector readRaw(void);

  void  setRange(qmc5883l_range_t range);
  qmc5883l_range_t getRange(void);

  void enableInterrupts();
  void disableInterrupts();

  void  setMeasurementMode(qmc5883l_mode_t mode);
  qmc5883l_mode_t getMeasurementMode(void);

  void  setDataRate(qmc5883l_dataRate_t dataRate);
  qmc5883l_dataRate_t getDataRate(void);

  void setOSR(qmc5883l_osr_t osr);
  qmc5883l_osr_t getOSR(void);

  private:

  float mgPerDigit;
  Vector v;

  void writeRegister8(uint8_t reg, uint8_t value);
  uint8_t readRegister8(uint8_t reg);
  uint8_t fastRegister8(uint8_t reg);
  int16_t readRegister16(uint8_t reg);
};

#endif