#if ARDUINO >= 100
#include "Arduino.h"
#else
#include "WProgram.h"
#endif

#include <Wire.h>
#include "QMC5883L.h"

bool QMC5883L::begin()
{

  Wire.begin();

  if ((fastRegister8(QMC5883L_REG_IDENT_B) != 0x00)
  || (fastRegister8(QMC5883L_REG_IDENT_C) != 0x01)
  || (fastRegister8(QMC5883L_REG_IDENT_D) != 0xFF))
  {
    return false;
  }

  setRange(QMC5883L_RANGE_8GA);
  setMeasurementMode(QMC5883L_CONTINOUS);
  setDataRate(QMC5883L_DATARATE_50HZ);
  setOSR(QMC5883L_OSR_64);

  /* Enable interrupts */
  enableInterrupts();

  mgPerDigit = 4.35f;

  return true;
    
}

Vector QMC5883L::readRaw(void)
{
  v.XAxis = readRegister16(QMC5883L_REG_OUT_X_L);
  v.YAxis = readRegister16(QMC5883L_REG_OUT_Y_L);
  v.ZAxis = readRegister16(QMC5883L_REG_OUT_Z_L);

  return v;
}

void QMC5883L::setRange(qmc5883l_range_t range)
{
    uint8_t value;

    switch(range)
    {
    case QMC5883L_RANGE_2GA:
      mgPerDigit = 1.22f;
      break;

    case QMC5883L_RANGE_8GA:
      mgPerDigit = 4.35f;
      break;

    default:
      break;
    }
    value = readRegister8(QMC5883L_REG_CONTROL_1);
    value &= 0xcf;
    value |= range;

    writeRegister8(QMC5883L_REG_CONTROL_1, value);
}

qmc5883l_range_t QMC5883L::getRange(void)
{
  uint8_t value;
  
  value = readRegister8(QMC5883L_REG_CONTROL_1);
  value &= 0b00110000;

  return (qmc5883l_range_t)((value >> 4));
}

void QMC5883L::disableInterrupts()
{
  uint8_t value;

  value = readRegister8(QMC5883L_REG_CONTROL_2);
  value |= 0x01;

  writeRegister8(QMC5883L_REG_CONTROL_2, value);
}

void QMC5883L::enableInterrupts()
{
  uint8_t value;

  value = readRegister8(QMC5883L_REG_CONTROL_2);
  value &= 0xfe;

  writeRegister8(QMC5883L_REG_CONTROL_2, value);
}

void QMC5883L::setMeasurementMode(qmc5883l_mode_t mode)
{
  uint8_t value;

  value = readRegister8(QMC5883L_REG_CONTROL_1);
  value &= 0xfc;
  value |= mode;

  writeRegister8(QMC5883L_REG_CONTROL_1, value);
}

qmc5883l_mode_t QMC5883L::getMeasurementMode(void)
{
  uint8_t value=0;

  value = readRegister8(QMC5883L_REG_CONTROL_1); 
  value &= 0b00000011;  

  return (qmc5883l_mode_t)value;
}

void QMC5883L::setDataRate(qmc5883l_dataRate_t dataRate)
{
  uint8_t value;

  value = readRegister8(QMC5883L_REG_CONTROL_1);
  value &= 0xf3;
  value |= (dataRate << 2);

  writeRegister8(QMC5883L_REG_CONTROL_1, value);
}

qmc5883l_dataRate_t QMC5883L::getDataRate(void)
{
  uint8_t value=0;

  value = readRegister8(QMC5883L_REG_CONTROL_1);
  value &= 0b00001100;
  value >>= 2;

  return (qmc5883l_dataRate_t)value;
}

void QMC5883L::setOSR(qmc5883l_osr_t osr)
{
  uint8_t value;

  value = readRegister8(QMC5883L_REG_CONTROL_1);
  value &= 0x3f;
  value |= (osr << 6);

  writeRegister8(QMC5883L_REG_CONTROL_1, value);
}

qmc5883l_osr_t QMC5883L::getOSR(void)
{
  uint8_t value=0;

  value = readRegister8(QMC5883L_REG_CONTROL_1);
  value &= 0b11000000;
  value >>= 6;

  return (qmc5883l_osr_t)value;
}

// Write byte to register
void QMC5883L::writeRegister8(uint8_t reg, uint8_t value)
{
	Wire.beginTransmission(QMC5883L_ADDRESS);
  #if ARDUINO >= 100
      Wire.write(reg);
      Wire.write(value);
  #else
      Wire.send(reg);
      Wire.send(value);
  #endif
  Wire.endTransmission();
}

// Read byte to register
uint8_t QMC5883L::fastRegister8(uint8_t reg)
{
  uint8_t value=0;
	Wire.beginTransmission(QMC5883L_ADDRESS);
  #if ARDUINO >= 100
    Wire.write(reg);
  #else
    Wire.send(reg);
  #endif
  Wire.endTransmission();

  Wire.requestFrom(QMC5883L_ADDRESS, 1);
  #if ARDUINO >= 100
      value = Wire.read();
  #else
      value = Wire.receive();
  #endif
  Wire.endTransmission();

 	return value;
}

// Read byte from register
uint8_t QMC5883L::readRegister8(uint8_t reg)
{
  uint8_t value=0;
  Wire.beginTransmission(QMC5883L_ADDRESS);
  #if ARDUINO >= 100
      Wire.write(reg);
  #else
      Wire.send(reg);
  #endif
  Wire.endTransmission();

  Wire.beginTransmission(QMC5883L_ADDRESS);
  Wire.requestFrom(QMC5883L_ADDRESS, 1);
  while(!Wire.available()) {};
  #if ARDUINO >= 100
      value = Wire.read();
  #else
      value = Wire.receive();
  #endif
  Wire.endTransmission();

  return value;
}

// Read word from register
int16_t QMC5883L::readRegister16(uint8_t reg)
{
  int16_t value=0;
  Wire.beginTransmission(QMC5883L_ADDRESS);
  #if ARDUINO >= 100
      Wire.write(reg);
  #else
      Wire.send(reg);
  #endif
  Wire.endTransmission();

  Wire.beginTransmission(QMC5883L_ADDRESS);
  Wire.requestFrom(QMC5883L_ADDRESS, 2);
  while(!Wire.available()) {};
  #if ARDUINO >= 100
      uint8_t vla = Wire.read();
      uint8_t vha = Wire.read();
  #else
      uint8_t vla = Wire.receive();
      uint8_t vha = Wire.receive();
  #endif
  Wire.endTransmission();

  value = (int16_t)((vha << 8) | vla);
 
  return value;
}

