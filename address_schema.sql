CREATE TABLE IF NOT EXISTS addresses (
    address_id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    source_raw_address TEXT NOT NULL,
    normalized_full_address TEXT NOT NULL,
    house_no VARCHAR(64) NULL,
    flat_no VARCHAR(64) NULL,
    floor VARCHAR(64) NULL,
    house_name VARCHAR(255) NULL,
    apartment_name VARCHAR(255) NULL,
    landmark VARCHAR(255) NULL,
    street VARCHAR(255) NULL,
    area VARCHAR(255) NULL,
    town VARCHAR(255) NULL,
    village VARCHAR(255) NULL,
    district VARCHAR(255) NULL,
    state VARCHAR(100) NULL,
    country VARCHAR(100) NOT NULL DEFAULT 'India',
    pincode VARCHAR(10) NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (address_id),
    KEY idx_addresses_pincode (pincode),
    KEY idx_addresses_state_district (state, district),
    KEY idx_addresses_town_area (town, area),
    FULLTEXT KEY ftx_normalized_full_address (normalized_full_address)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS feedback_events (
    feedback_id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    address_id BIGINT UNSIGNED NULL,
    arm INT NOT NULL,
    accepted TINYINT(1) NOT NULL,
    reward FLOAT NOT NULL,
    metadata JSON NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (feedback_id),
    KEY idx_feedback_address_id (address_id),
    CONSTRAINT fk_feedback_address
        FOREIGN KEY (address_id) REFERENCES addresses(address_id)
        ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
