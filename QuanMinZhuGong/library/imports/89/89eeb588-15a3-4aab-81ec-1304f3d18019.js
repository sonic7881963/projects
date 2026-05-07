"use strict";
cc._RF.push(module, '89eebWIFaNKq4HsEwTz0YAZ', 'left - 004');
// 3.16小游戏/command_TypeScript/level8/left - 004.ts

"use strict";
var __extends = (this && this.__extends) || (function () {
    var extendStatics = function (d, b) {
        extendStatics = Object.setPrototypeOf ||
            ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
            function (d, b) { for (var p in b) if (Object.prototype.hasOwnProperty.call(b, p)) d[p] = b[p]; };
        return extendStatics(d, b);
    };
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
var __decorate = (this && this.__decorate) || function (decorators, target, key, desc) {
    var c = arguments.length, r = c < 3 ? target : desc === null ? desc = Object.getOwnPropertyDescriptor(target, key) : desc, d;
    if (typeof Reflect === "object" && typeof Reflect.decorate === "function") r = Reflect.decorate(decorators, target, key, desc);
    else for (var i = decorators.length - 1; i >= 0; i--) if (d = decorators[i]) r = (c < 3 ? d(r) : c > 3 ? d(target, key, r) : d(target, key)) || r;
    return c > 3 && r && Object.defineProperty(target, key, r), r;
};
Object.defineProperty(exports, "__esModule", { value: true });
// Learn TypeScript:
//  - https://docs.cocos.com/creator/manual/en/scripting/typescript.html
// Learn Attribute:
//  - https://docs.cocos.com/creator/manual/en/scripting/reference/attributes.html
// Learn life-cycle callbacks:
//  - https://docs.cocos.com/creator/manual/en/scripting/life-cycle-callbacks.html
var global___004_1 = require("./global - 004");
var _a = cc._decorator, ccclass = _a.ccclass, property = _a.property;
var mainCharacter = /** @class */ (function (_super) {
    __extends(mainCharacter, _super);
    function mainCharacter() {
        var _this = _super !== null && _super.apply(this, arguments) || this;
        _this.label = null;
        _this.text = 'hello';
        return _this;
    }
    // LIFE-CYCLE CALLBACKS:
    mainCharacter.prototype.onLoad = function () {
        var manager = cc.director.getCollisionManager();
        manager.enabled = true;
    };
    //产生碰撞会调用
    mainCharacter.prototype.onCollisionEnter = function (other, self) {
        cc.log("开始碰撞" + other.tag);
        if (global___004_1.default.attack == true) {
            global___004_1.default.minion1_hp -= 999;
            var damage = cc.find("Canvas/bj/kuan/main_damage");
            damage.getComponent(cc.Label).string = "-999";
            var damage2 = cc.find("Canvas/bj/kuan/enemy_damage");
            damage2.getComponent(cc.Label).string = "";
        }
        global___004_1.default.attack = false;
    };
    mainCharacter.prototype.onCollisionStay = function (other) {
        global___004_1.default.attack = false;
        global___004_1.default.minion_attack = false;
    };
    mainCharacter.prototype.onCollisionExit = function (other, self) {
        cc.log("碰撞结束");
        if (global___004_1.default.minion1_hp <= 0) {
            other.node.active = false;
            global___004_1.default.attack = false;
            global___004_1.default.minion_attack = false;
            var damage = cc.find("Canvas/bj/kuan/main_damage");
            damage.getComponent(cc.Label).string = "";
            var damage2 = cc.find("Canvas/bj/kuan/enemy_damage");
            damage2.getComponent(cc.Label).string = "";
            var node1 = cc.find("Canvas/bj/d-8/c-8");
            node1.active = true;
            var node2 = cc.find("Canvas/bj/b/a");
            node2.active = true;
            self.node.active = false;
            var node3 = cc.find("Canvas/bj/kuan/tgcg");
            node3.active = true;
        }
        else {
            other.node.children[0].setContentSize(global___004_1.default.minion1_hp, 19);
        }
    };
    mainCharacter.prototype.start = function () {
        this.main_x = this.node.position.x;
    };
    mainCharacter.prototype.update = function (dt) {
        if (global___004_1.default.attack == true) {
            this.node.setPosition(this.node.position.x + 1000 * dt, this.node.position.y);
        }
        else {
            if (!(this.node.position.x <= this.main_x + 50) && (this.node.position.x >= this.main_x - 50)) {
                this.node.setPosition(this.node.position.x - 1000 * dt, this.node.position.y);
            }
        }
    };
    __decorate([
        property(cc.Label)
    ], mainCharacter.prototype, "label", void 0);
    __decorate([
        property
    ], mainCharacter.prototype, "text", void 0);
    mainCharacter = __decorate([
        ccclass
    ], mainCharacter);
    return mainCharacter;
}(cc.Component));
exports.default = mainCharacter;

cc._RF.pop();