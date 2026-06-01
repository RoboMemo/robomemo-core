/**
 * AES-256-GCM Data Encryption Service
 * 
 * Provides at-rest encryption for sensitive data fields.
 * Master key loaded from environment variable.
 */

const crypto = require('crypto');

const ALGORITHM = 'aes-256-gcm';
const IV_LENGTH = 16;
const TAG_LENGTH = 16;

class DataEncryption {
  constructor() {
    const keyHex = process.env.ENCRYPTION_MASTER_KEY;
    if (keyHex && keyHex.length === 64) {
      this.key = Buffer.from(keyHex, 'hex');
      this.enabled = true;
    } else {
      console.warn('[Encryption] ENCRYPTION_MASTER_KEY not set or invalid (need 64 hex chars). Encryption disabled.');
      this.enabled = false;
    }
  }

  /**
   * Encrypt plaintext → { ciphertext, iv, tag } (all hex strings)
   */
  encrypt(plaintext) {
    if (!this.enabled) return { ciphertext: plaintext, iv: '', tag: '', encrypted: false };
    const iv = crypto.randomBytes(IV_LENGTH);
    const cipher = crypto.createCipheriv(ALGORITHM, this.key, iv);
    let encrypted = cipher.update(plaintext, 'utf8', 'hex');
    encrypted += cipher.final('hex');
    return {
      ciphertext: encrypted,
      iv: iv.toString('hex'),
      tag: cipher.getAuthTag().toString('hex'),
      encrypted: true,
    };
  }

  /**
   * Decrypt → plaintext
   */
  decrypt(ciphertext, ivHex, tagHex) {
    if (!this.enabled || !ivHex || !tagHex) return ciphertext;
    const iv = Buffer.from(ivHex, 'hex');
    const tag = Buffer.from(tagHex, 'hex');
    const decipher = crypto.createDecipheriv(ALGORITHM, this.key, iv);
    decipher.setAuthTag(tag);
    let decrypted = decipher.update(ciphertext, 'hex', 'utf8');
    decrypted += decipher.final('utf8');
    return decrypted;
  }

  /**
   * Convenience: encrypt a JSON-serializable object
   */
  encryptJSON(obj) {
    return this.encrypt(JSON.stringify(obj));
  }

  /**
   * Convenience: decrypt to a parsed JS object
   */
  decryptJSON(ciphertext, ivHex, tagHex) {
    const plain = this.decrypt(ciphertext, ivHex, tagHex);
    try { return JSON.parse(plain); } catch { return plain; }
  }

  isEnabled() { return this.enabled; }

  /**
   * Generate a new random 256-bit master key (for initial setup)
   */
  static generateMasterKey() {
    return crypto.randomBytes(32).toString('hex');
  }
}

module.exports = new DataEncryption();
module.exports.DataEncryption = DataEncryption;
