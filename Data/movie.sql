-- MySQL Script generated by MySQL Workbench
-- Fri Sep 29 11:42:15 2023
-- Model: New Model    Version: 1.0
-- MySQL Workbench Forward Engineering

SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0;
SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0;
SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='ONLY_FULL_GROUP_BY,STRICT_TRANS_TABLES,NO_ZERO_IN_DATE,NO_ZERO_DATE,ERROR_FOR_DIVISION_BY_ZERO,NO_ENGINE_SUBSTITUTION';

-- -----------------------------------------------------
-- Schema movie
-- -----------------------------------------------------
DROP SCHEMA IF EXISTS `movie` ;

-- -----------------------------------------------------
-- Schema movie
-- -----------------------------------------------------
CREATE SCHEMA IF NOT EXISTS `movie` DEFAULT CHARACTER SET utf8 ;
USE `movie` ;

-- -----------------------------------------------------
-- Table `movie`.`title_basics`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `movie`.`title_basics` ;

CREATE TABLE IF NOT EXISTS `movie`.`title_basics` (
  `tconst` VARCHAR(45) NOT NULL,
  `primary_title` VARCHAR(300) NULL,
  `start_year` INT NULL,
  `runtime` INT NULL,
  PRIMARY KEY (`tconst`))
ENGINE = InnoDB;


-- -----------------------------------------------------
-- Table `movie`.`ratings`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `movie`.`ratings` ;

CREATE TABLE IF NOT EXISTS `movie`.`ratings` (
  `tconst` VARCHAR(45) NOT NULL,
  `average_rating` DECIMAL(10,2) NULL,
  `number_of_votes` INT NULL,
  PRIMARY KEY (`tconst`))
ENGINE = InnoDB;


-- -----------------------------------------------------
-- Table `movie`.`genres`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `movie`.`genres` ;

CREATE TABLE IF NOT EXISTS `movie`.`genres` (
  `genre_id` INT NOT NULL,
  `genre_name` VARCHAR(45) NULL,
  PRIMARY KEY (`genre_id`))
ENGINE = InnoDB;


-- -----------------------------------------------------
-- Table `movie`.`title_genres`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `movie`.`title_genres` ;

CREATE TABLE IF NOT EXISTS `movie`.`title_genres` (
  `tconst` VARCHAR(45) NOT NULL,
  `genre_id` INT NOT NULL)
ENGINE = InnoDB;


SET SQL_MODE=@OLD_SQL_MODE;
SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS;
SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS;
